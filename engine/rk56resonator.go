package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

type RK56resonator struct {
}

func (rk *RK56resonator) Step() {

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

	k1, k2, k3, k4, k5, k6, k7, k8 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	defer cuda.Recycle(k7)
	defer cuda.Recycle(k8)
	//k2 will be recyled as k9

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL
	Dt32:=float32(Dt_si)	

	// stage 1
	torqueFn(k1)
	k1i,k1v:=SetResonatorFnNew()

	// stage 2
	Time = t0 + (1./6.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
	M.normalize()
	I+=k1i*(1./6.)*Dt32
	V+=k1v*(1./6.)*Dt32
	torqueFn(k2)
	k2i,k2v:=SetResonatorFnNew()

	// stage 3
	Time = t0 + (4./15.)*Dt_si
	cuda.Madd3(m, m0, k1, k2, 1, (4./75.)*h, (16./75.)*h)
	M.normalize()
	I=I0+k1i*(4./75.)*Dt32+k2i*(16./75.)*Dt32
	V=V0+k1v*(4./75.)*Dt32+k2v*(16./75.)*Dt32
	torqueFn(k3)
	k3i,k3v:=SetResonatorFnNew()	

	// stage 4
	Time = t0 + (2./3.)*Dt_si
	madd4(m, m0, k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
	M.normalize()
	I=I0+k1i*(5./6.)*Dt32+k2i*(-8./3.)*Dt32+k3i*(5./2.)*Dt32
	V=V0+k1v*(5./6.)*Dt32+k2v*(-8./3.)*Dt32+k3v*(5./2.)*Dt32
	torqueFn(k4)
	k4i,k4v:=SetResonatorFnNew()

	// stage 5
	Time = t0 + (4./5.)*Dt_si
	madd5(m, m0, k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
	M.normalize()
	I=I0+k1i*(-8./5.)*Dt32+k2i*(144./25.)*Dt32+k3i*(-4.)*Dt32+k4i*(16./25.)*Dt32
	V=V0+k1v*(-8./5.)*Dt32+k2v*(144./25.)*Dt32+k3v*(-4.)*Dt32+k4v*(16./25.)*Dt32
	torqueFn(k5)
	k5i,k5v:=SetResonatorFnNew()

	// stage 6
	Time = t0 + (1.)*Dt_si
	madd6(m, m0, k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
	M.normalize()
	I=I0+k1i*(361./320.)*Dt32+k2i*(-18./5.)*Dt32+k3i*(407./128.)*Dt32+k4i*(-11./80.)*Dt32+k5i*(55./128.)*Dt32
	V=V0+k1v*(361./320.)*Dt32+k2v*(-18./5.)*Dt32+k3v*(407./128.)*Dt32+k4v*(-11./80.)*Dt32+k5v*(55./128.)*Dt32
	torqueFn(k6)

	// stage 7
	Time = t0
	madd5(m, m0, k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
	M.normalize()
	I=I0+k1i*(-11./640.)*Dt32+k3i*(11./256.)*Dt32+k4i*(-11./160.)*Dt32+k5i*(11./256.)*Dt32
	V=V0+k1v*(-11./640.)*Dt32+k3v*(11./256.)*Dt32+k4v*(-11./160.)*Dt32+k5v*(11./256.)*Dt32
	torqueFn(k7)
	k7i,k7v:=SetResonatorFnNew()

	// stage 8
	Time = t0 + (1.)*Dt_si
	madd7(m, m0, k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
	M.normalize()
	I=I0+k1i*(93./640.)*Dt32+k2i*(-18./5.)*Dt32+k3i*(803./256.)*Dt32+k4i*(-11./160.)*Dt32+k5i*(99./256.)*Dt32+k7i*(1.)*Dt32
	V=V0+k1v*(93./640.)*Dt32+k2v*(-18./5.)*Dt32+k3v*(803./256.)*Dt32+k4v*(-11./160.)*Dt32+k5v*(99./256.)*Dt32+k7v*(1.)*Dt32
	torqueFn(k8)
	k8i,k8v:=SetResonatorFnNew()

	// stage 9: 6th order solution
	Time = t0 + (1.)*Dt_si
	//madd6(m, m0, k1, k3, k4, k5, k6, 1, (31./384.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h)
	madd7(m, m0, k1, k3, k4, k5, k7, k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
	M.normalize()
	I=I0+k1i*(7./1408.)*Dt32+k3i*(1125./2816.)*Dt32+k4i*(9./32.)*Dt32+k5i*(125./768.)*Dt32+k7i*(5./66.)*Dt32+k8i*(5./66.)*Dt32
	V=V0+k1v*(7./1408.)*Dt32+k3v*(1125./2816.)*Dt32+k4v*(9./32.)*Dt32+k5v*(125./768.)*Dt32+k7v*(5./66.)*Dt32+k8v*(5./66.)*Dt32
	torqueFn(k2) // re-use k2

	// error estimate
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)
	madd4(Err, k1, k6, k7, k8, (-5. / 66.), (-5. / 66.), (5. / 66.), (5. / 66.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k2)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./7.))
	}
}

func (rk *RK56resonator) Free() {
}

