package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for addresonatorfield kernel
var addresonatorfield_code cu.Function

// Stores the arguments for addresonatorfield kernel invocation
type addresonatorfield_args_t struct{
	 arg_Bx unsafe.Pointer
	 arg_By unsafe.Pointer
	 arg_Bz unsafe.Pointer
	 arg_voltage float32
	 arg_current float32
	 arg_brf float32
	 arg_N int
	 argptr [7]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addresonatorfield kernel invocation
var addresonatorfield_args addresonatorfield_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 addresonatorfield_args.argptr[0] = unsafe.Pointer(&addresonatorfield_args.arg_Bx)
	 addresonatorfield_args.argptr[1] = unsafe.Pointer(&addresonatorfield_args.arg_By)
	 addresonatorfield_args.argptr[2] = unsafe.Pointer(&addresonatorfield_args.arg_Bz)
	 addresonatorfield_args.argptr[3] = unsafe.Pointer(&addresonatorfield_args.arg_voltage)
	 addresonatorfield_args.argptr[4] = unsafe.Pointer(&addresonatorfield_args.arg_current)
	 addresonatorfield_args.argptr[5] = unsafe.Pointer(&addresonatorfield_args.arg_brf)
	 addresonatorfield_args.argptr[6] = unsafe.Pointer(&addresonatorfield_args.arg_N)
	 }

// Wrapper for addresonatorfield CUDA kernel, asynchronous.
func k_addresonatorfield_async ( Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, voltage float32, current float32, brf float32, N int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("addresonatorfield")
	}

	addresonatorfield_args.Lock()
	defer addresonatorfield_args.Unlock()

	if addresonatorfield_code == 0{
		addresonatorfield_code = fatbinLoad(addresonatorfield_map, "addresonatorfield")
	}

	 addresonatorfield_args.arg_Bx = Bx
	 addresonatorfield_args.arg_By = By
	 addresonatorfield_args.arg_Bz = Bz
	 addresonatorfield_args.arg_voltage = voltage
	 addresonatorfield_args.arg_current = current
	 addresonatorfield_args.arg_brf = brf
	 addresonatorfield_args.arg_N = N
	

	args := addresonatorfield_args.argptr[:]
	cu.LaunchKernel(addresonatorfield_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("addresonatorfield")
	}
}

// maps compute capability on PTX code for addresonatorfield kernel.
var addresonatorfield_map = map[int]string{ 0: "" ,
30: addresonatorfield_ptx_30 ,
35: addresonatorfield_ptx_35 ,
37: addresonatorfield_ptx_37 ,
50: addresonatorfield_ptx_50 ,
52: addresonatorfield_ptx_52 ,
53: addresonatorfield_ptx_53 ,
60: addresonatorfield_ptx_60 ,
61: addresonatorfield_ptx_61 ,
70: addresonatorfield_ptx_70 ,
75: addresonatorfield_ptx_75  }

// addresonatorfield PTX code for various compute capabilities.
const(
  addresonatorfield_ptx_30 = `
.version 6.4
.target sm_30
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_35 = `
.version 6.4
.target sm_35
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_37 = `
.version 6.4
.target sm_37
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_50 = `
.version 6.4
.target sm_50
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_52 = `
.version 6.4
.target sm_52
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_53 = `
.version 6.4
.target sm_53
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_60 = `
.version 6.4
.target sm_60
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_61 = `
.version 6.4
.target sm_61
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_70 = `
.version 6.4
.target sm_70
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
   addresonatorfield_ptx_75 = `
.version 6.4
.target sm_75
.address_size 64

	// .globl	addresonatorfield

.visible .entry addresonatorfield(
	.param .u64 addresonatorfield_param_0,
	.param .u64 addresonatorfield_param_1,
	.param .u64 addresonatorfield_param_2,
	.param .f32 addresonatorfield_param_3,
	.param .f32 addresonatorfield_param_4,
	.param .f32 addresonatorfield_param_5,
	.param .u32 addresonatorfield_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [addresonatorfield_param_0];
	ld.param.f32 	%f1, [addresonatorfield_param_4];
	ld.param.f32 	%f2, [addresonatorfield_param_5];
	ld.param.u32 	%r2, [addresonatorfield_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	st.global.f32 	[%rd4], %f4;

BB0_2:
	ret;
}


`
 )
