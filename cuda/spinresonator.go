package cuda

import(
	"github.com/mumax/3/data"
)

//Add resonator field to Beff.
//see spinresonatorfield.cu
//After compilation by typing "make" in cuda/ in command line, cuda will automatically generate spinresonator_wrapper.go
//In spinresonator_wrapper.go, there is a function k_addresonatorfield_async
 
func AddResonatorField(Beff *data.Slice,voltage, current, brf float32){

	N:=Beff.Len()
	cfg:=make1DConf(N)

	k_addresonatorfield_async(
		Beff.DevPtr(X),Beff.DevPtr(Y),Beff.DevPtr(Z),
		voltage,current,brf,N,cfg)
}
