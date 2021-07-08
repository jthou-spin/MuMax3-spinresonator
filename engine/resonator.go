package engine

import (
//	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
//	"unsafe"
	"math"
	"math/rand"
	"time"
)

//All variables here are global, because they are first letter capital. V and I will be updated if solver involves SetResonatorFnNew
//Other variables, such as L,C,Brf,and R, are treated as time independent parameters
//variables in float32type, which is similar to Slice for magnetization data. This is due to cuda programming requirement.
var(
	V	float32=float32(0.0)		// Resonator voltage(in V), use 32 because operations in cuda need 32, not float64
	I	float32=float32(0.0)		// Resonator current(in A)
	L	float32=float32(1.0)		// Resonator inductor(in H)
	C	float32=float32(1.0)		// Resonator capacitor(in F)
	Brf	float32=float32(1.0) 		// Magnetic field per inductive current (T/Ampere)
	Brfini  float32=float32(1.0)		// Records the initial value of Brfini
	Brfchange float32=float32(5)		// Chaneg of Brf, only used for rk56resonatortimedepBrf
	Tini   float32=float32(2e-7)		// Initial time that Brf starts to change linearly, only used for rk56resonatortimedepBrf
	Tfinal  float32=float32(12e-7)		// Final time that Brf stops to change linearly, only used for rk56resonatortimedepBrf
	R       float32=float32(1.0)		// Resistor (Ohm)
	TempRes	float32=float32(0.0)
	V_therm	thermResonator
	I_therm thermResonator
//	V_go	float32
//	I_go 	float32
	Vout	=NewScalarValue("Vout","Volt","Resonator Voltage",GetVoltage) //Use tableadd(Vout) to print V, just as E_total in engine/energy.go
	Iout    =NewScalarValue("Iout","Ampere","Resonator Current",GetCurrent) //Use tableadd(Iout) to print I, just as E_total
	Lout    =NewScalarValue("Lout","Henry","Resonator Inductor",GetInductor) //Use tableadd(Lout) to print L, just as E_total
	Cout    =NewScalarValue("Cout","Farad","Resonator Capacitor",GetCapacitor) //Use tableadd(Cout) to print C, just as E_total
	Brfout  =NewScalarValue("Brfout","T/A","Resonator Coupling",GetBrf)
)

type thermResonator struct{
//	seed		int64
//	generator	curand.Generator
//	noise 		*float32
//	noise		[]unsafe.Pointer	
//	noise		unsafe.Pointer
	noise		float64
	step		int
	dt		float64
}

//Setting the variables using function calls in the script, such as InitialV(initialvoltage)
func init() {
	DeclFunc("InitialV",InitialV,"Set the initial value of V")
	DeclFunc("InitialI",InitialI,"Set the initial value of I")
	DeclFunc("SetL",SetL,"Set the value of L")
	DeclFunc("SetC",SetC,"Set the value of C")
	DeclFunc("SetBrf",SetBrf,"Set the value of Brf")
	DeclFunc("SetR",SetR,"Set the value of R")
	DeclFunc("SetBrfchange",SetBrfchange,"Set Brfchange")
	DeclFunc("SetTini",SetTini,"Set Tini")
	DeclFunc("SetTfinal",SetTfinal,"Set Tfinal")
	DeclFunc("SetTempRes",SetTempRes,"Set Resonator Temperature")
//	DeclFunc("ThermISeed",ThermISeed,"Set a random seed for I thermal noise")
//	DeclFunc("ThermVSeed",ThermVSeed,"Set a random seed for V thermal noise")
	I_therm.step=-1
	V_therm.step=-1
	I_therm.dt=-1
	V_therm.dt=-1
	DeclROnly("I_therm",&I_therm,"I Thermal Noise (A)")
	DeclROnly("V_therm",&V_therm,"V Thermal Noise (V)")
}

//The values in script are float64type, but variables in the program is float32type.
func InitialV(voltagevalue float64){
	V=float32(voltagevalue)
}

func InitialI(currentvalue float64){
	I=float32(currentvalue)
}

func SetL(inductorvalue float64){
	L=float32(inductorvalue)
}

func SetC(capacitorvalue float64){
	C=float32(capacitorvalue)
}

func SetBrf(brfvalue float64){
	Brf=float32(brfvalue)
	Brfini=float32(brfvalue)
}

func SetR(resistorvalue float64){
	R=float32(resistorvalue)
}

func SetBrfchange(brfchangevalue float64){
	Brfchange=float32(brfchangevalue)
}

func SetTini(tinivalue float64){
	Tini=float32(tinivalue)
}

func SetTfinal(tfinalvalue float64){
	Tfinal=float32(tfinalvalue)
}

func SetTempRes(tempresvalue float64){
	TempRes=float32(tempresvalue)
}


//Msat in unit of A/m,cellvolume in m^3, inductor in Henry 
//TotalMoment returns the total magnetic moment xyz in the simulation problem
func SetResonatorFnNew() (float32, float32){
	totalmagneticmoment:=TotalMoment()
	MX:=float32(totalmagneticmoment[0])
	forcurrent:=float32(V/L-R*I/L)
	forvoltage:=float32(-I/C+Brf*MX/L/C)
	return forcurrent,forvoltage
}


func SetResonatorFnNewNew(ktorque *data.Slice) (float32, float32){
	m := M.Buffer()
	size := m.Size()
	mscaletorque := cuda.Buffer(3, size)
	defer cuda.Recycle(mscaletorque)

	msat,rM := Msat.Slice()
	if rM{
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++{
		cuda.Mul(mscaletorque.Comp(c), ktorque.Comp(c), msat)
	}
	totaltorque:=make([]float32,3)
	cellvolume:=lazy_cellsize[0]*lazy_cellsize[1]*lazy_cellsize[2]
	for i:=range totaltorque{
		totaltorque[i]=float32(cuda.Sum(mscaletorque.Comp(i)))*float32(cellvolume)*float32(GammaLL)
	}
	forcurrent:=float32(V/L-R*I/L-Brf*totaltorque[0]/L)
	forvoltage:=float32(-I/C)
	if !Temp.isZero(){
		V_therm.Vupdate()
		I_therm.Iupdate()
//		forcurrent=forcurrent+*(I_therm.noise)
//		forvoltage=forvoltage+*(V_therm.noise)			
//		forcurrent=forcurrent+I_therm.noise[0]
//		forvoltage=forvoltage+V_therm.noise[0]
		forcurrent=forcurrent+float32(I_therm.noise)
		forvoltage=forvoltage+float32(V_therm.noise)
//		forcurrent=forcurrent+float32(I_go)
//		forvoltage=forvoltage+float32(V_go)
	}
	return forcurrent,forvoltage
}

//Please see function SetMFull in engine/demag.go
//I use similar codes from there, but with multiplication of cell size 
func TotalMoment() []float32{
	m := M.Buffer()
	size := m.Size()
	mScaleMsat := cuda.Buffer(3, size)
	defer cuda.Recycle(mScaleMsat)

	msat,rM := Msat.Slice()
	if rM{
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++{
		cuda.Mul(mScaleMsat.Comp(c), M.Buffer().Comp(c), msat)
	}
	totalmoment:=make([]float32,3)
	cellvolume:=lazy_cellsize[0]*lazy_cellsize[1]*lazy_cellsize[2]
	for i:=range totalmoment{
		totalmoment[i]=float32(cuda.Sum(mScaleMsat.Comp(i)))*float32(cellvolume)
	}
	return totalmoment
}

//func TotalTorque(ktorque *data.Slice) []float32{
//	mscaletorque := cuda.Buffer(3, size)
//	defer cuda.Recycle(mscaletorque)
//
//	msat,rM := Msat.Slice()
//	if rM{
//		defer cuda.Recycle(msat)
//	}
//	for c := 0; c < 3; c++{
//		cuda.Mul(mscalemtorque.Comp(c), ktorque.Comp(c), msat)
//	}
//	totaltorque:=make([]float32,3)
//	cellvolume:=lazy_cellsize[0]*lazy_cellsize[1]*lazy_cellsize[2]
//	for i:=range totaltorque{
//		totaltorque[i]=float32(cuda.Sum(mscaletorque.Comp(i)))*float32(cellvolume)
//	}
//	return totaltorque
//}

func (res *thermResonator) Iupdate(){
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.
	if FixDt != 0 {
		Dt_si = FixDt
	}

//	if res.generator == 0 {
//		res.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
//		res.generator.SetSeed(res.seed)
//	}
//	if res.noise == nil {
//		res.noise = make([]float32,1)
//		res.noise = make([]unsafe.Pointer,1)
//		res.noise = unsafe.Pointer(&make([]float32,1)[0])
//		r:= rand.New(rand.NewSource(time.Now().UnixNano()))	
//		res.noise = r.NormFloat64()
//		// when noise was (re-)allocated it's invalid for sure.
//		I_therm.step = -1
//		I_therm.dt = -1
//	}

	if Temp.isZero() {
		res.step = NSteps
		res.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == res.step && Dt_si == res.dt {
		return
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == res.step && Dt_si != res.dt {
	//	*res.noise=(*res.noise)*float32(math.Sqrt(res.dt/Dt_si))	
	//	res.noise[0]=res.noise[0]*float32(math.Sqrt(res.dt/Dt_si))	
	//	*(*float32)(res.noise)=*(*float32)(res.noise)*float32(math.Sqrt(res.dt/Dt_si))
	//	r:= rand.New(rand.NewSource(time.Now().UnixNano()))	
		res.noise=res.noise*math.Sqrt(res.dt/Dt_si)
		res.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}
	
	dt32:=float32(Dt_si)	
	const mean = 0
	const stddev = 1
//	res.generator.GenerateNormal(uintptr(res.noise), int64(1), mean, stddev)
//	*res.noise=(*res.noise)*float32(math.Sqrt(float64(R*mag.Kb*TempRes/(L*L*dt32))))
//	*(*float32)(res.noise)=*(*float32)(res.noise)*float32(math.Sqrt(float64(R*mag.Kb*TempRes/(L*L*dt32))))
	r:= rand.New(rand.NewSource(time.Now().UnixNano()))	
	res.noise = r.NormFloat64()
	res.noise=res.noise*math.Sqrt(float64(R*mag.Kb*TempRes/(L*L*dt32)))
	res.step = NSteps
	res.dt = Dt_si
}


func (res *thermResonator) Vupdate(){
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.
	if FixDt != 0 {
		Dt_si = FixDt
	}

//	if res.generator == 0 {
//		res.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
//		res.generator.SetSeed(res.seed)
//	}
//	if res.noise == nil {
//	//	res.noise = make([]float32,1)
//	//	res.noise = make([]unsafe.Pointer,1)
//		res.noise = unsafe.Pointer(&make([]float32,1)[0])
//		// when noise was (re-)allocated it's invalid for sure.
//		V_therm.step = -1
//		V_therm.dt = -1
//	}

	if Temp.isZero() {
		res.step = NSteps
		res.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == res.step && Dt_si == res.dt {
		return
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == res.step && Dt_si != res.dt {
	//	*res.noise=(*res.noise)*float32(math.Sqrt(res.dt/Dt_si))	
	//	res.noise[0]=res.noise[0]*float32(math.Sqrt(res.dt/Dt_si))	
	//	*(*float32)(res.noise)=*(*float32)(res.noise)*float32(math.Sqrt(res.dt/Dt_si))
		res.noise=res.noise*math.Sqrt(res.dt/Dt_si)
		res.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}
	
	dt32:=float32(Dt_si)
	const mean = 0
	const stddev = 1
//	res.generator.GenerateNormal(uintptr(res.noise), int64(1), mean, stddev)
//	*res.noise=(*res.noise)*float32(math.Sqrt(float64(R*mag.Kb*TempRes/(L*C*dt32))))
//	*(*float32)(res.noise)=*(*float32)(res.noise)*float32(math.Sqrt(float64(R*mag.Kb*TempRes/(L*C*dt32))))
	r:= rand.New(rand.NewSource(time.Now().UnixNano()))	
	res.noise = r.NormFloat64()
	res.noise=res.noise*math.Sqrt(float64(R*mag.Kb*TempRes/(L*C*dt32)))
	res.step = NSteps
	res.dt = Dt_si
}



//func ThermISeed(seed int){
//	I_therm.seed=int64(seed)
//	if I_therm.generator !=0 {
//		I_therm.generator.SetSeed(I_therm.seed)
//	}
//}


//func ThermVSeed(seed int){
//	V_therm.seed=int64(seed)
//	if V_therm.generator !=0 {
//		V_therm.generator.SetSeed(V_therm.seed)
//	}
//}






//Similar to GetTotalEnergy() in engine/energy.go
func GetVoltage() float64{
	return float64(V)	
}

func GetCurrent() float64{
	return float64(I)
}

func GetInductor() float64{
	return float64(L)
}

func GetCapacitor() float64{
	return float64(C)
}

func GetBrf() float64{
	return float64(Brf)
}
