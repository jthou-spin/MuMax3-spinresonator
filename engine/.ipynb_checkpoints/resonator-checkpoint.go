package engine

import (
	"github.com/mumax/3/cuda"
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
	R       float32=float32(1.0)		// Resistor (Ohm)
	Vout	=NewScalarValue("Vout","Volt","Resonator Voltage",GetVoltage) //Use tableadd(Vout) to print V, just as E_total in engine/energy.go
	Iout    =NewScalarValue("Iout","Ampere","Resonator Current",GetCurrent) //Use tableadd(Iout) to print I, just as E_total
	Lout    =NewScalarValue("Lout","Henry","Resonator Inductor",GetInductor) //Use tableadd(Lout) to print L, just as E_total
	Cout    =NewScalarValue("Cout","Farad","Resonator Capacitor",GetCapacitor) //Use tableadd(Cout) to print C, just as E_total
)

//Setting the variables using function calls in the script, such as InitialV(initialvoltage)
func init() {
	DeclFunc("InitialV",InitialV,"Set the initial value of V")
	DeclFunc("InitialI",InitialI,"Set the initial value of I")
	DeclFunc("SetL",SetL,"Set the value of L")
	DeclFunc("SetC",SetC,"Set the value of C")
	DeclFunc("SetBrf",SetBrf,"Set the value of Brf")
	DeclFunc("SetR",SetR,"Set the value of R")
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
}

func SetR(resistorvalue float64){
	R=float32(resistorvalue)
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
