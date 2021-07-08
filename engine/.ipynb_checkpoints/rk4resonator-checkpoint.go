package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Classical 4th order RK solver with resonator
type RK4resonator struct {
}

func (rk *RK4resonator) Step() {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	V0:=V
	I0:=I
	
	
	k1, k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL
	Dt32:=float32(Dt_si)

	// stage 1
	torqueFn(k1)
	k1i,k1v:=SetResonatorFnNew()	

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	M.normalize()
	I+=k1i*(1./2.)*Dt32
	V+=k1v*(1./2.)*Dt32
	torqueFn(k2)
	k2i,k2v:=SetResonatorFnNew()	


	// stage 3
	cuda.Madd2(m, m0, k2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	M.normalize()
	I=I0+k2i*(1./2.)*Dt32
	V=V0+k2v*(1./2.)*Dt32
	torqueFn(k3)
	k3i,k3v:=SetResonatorFnNew()	

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(m, m0, k3, 1, 1.*h) // m = m0*1 + k3*1
	M.normalize()
	I=I0+k3i*1.*Dt32
	V=V0+k3v*1.*Dt32
	torqueFn(k4)
	k4i,k4v:=SetResonatorFnNew()	

	err := cuda.MaxVecDiff(k1, k4) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		madd5(m, m0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
		M.normalize()
		I=I0+k1i*(1./6.)*Dt32+k2i*(1./3.)*Dt32+k3i*(1./3.)*Dt32+k4i*(1./6.)*Dt32
		V=V0+k1v*(1./6.)*Dt32+k2v*(1./3.)*Dt32+k3v*(1./3.)*Dt32+k4v*(1./6.)*Dt32
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
		setLastErr(err)
		setMaxTorque(k4)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	}
}

func (_ *RK4resonator) Free() {}
