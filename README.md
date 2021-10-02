mumax3 with spin-resonator coupling
======
[![Build Status](https://travis-ci.org/mumax/3.svg?branch=master)](https://travis-ci.org/mumax/3)

GPU accelerated micromagnetic simulator with spin-resonator coupling. The spin dynamics is coupled with an LCR resonator which represents the photon mode. The code is modified based on Mumax 3.10. 


Downloads and documentation
---------------------------

http://mumax.github.io


Paper
-----

The Design and Verification of mumax3:

http://scitation.aip.org/content/aip/journal/adva/4/10/10.1063/1.4899186

Proposal for a Spin-Torque-Oscillator Maser Enabled by Microwave Photon-Spin Coupling
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.16.034034

Tools
-----

https://godoc.org/github.com/mumax/3/cmd


Building from source
--------------------

Consider downloading a pre-compiled binary. If you want to compile nevertheless:

  * install the nvidia proprietary driver, if not yet present.
   - if unsure, it's probably already there
   - v340 recommended
  * install Go 
    - https://golang.org/dl/
    - set $GOPATH
  * install CUDA 
    - https://developer.nvidia.com/cuda-downloads (pick default installation path)
    - or `sudo apt-get install nvidia-cuda-toolkit`
  * install a C compiler
    - Ubuntu: `sudo apt-get install gcc`
    - MacOSX: https://developer.apple.com/xcode/download/
    - Windows: http://sourceforge.net/projects/mingw-w64/
  * if you have git installed: 
    - `go get github.com/jthou0129/Mumax3_spinresonator/cmd/mumax3`
  * if you don't have git:
    - seriously, no git?
    - get the source from https://github.com/mumax/3/releases
    - unzip the source into $GOPATH/src/github.com/mumax/3
    - `cd $GOPATH/src/github.com/jthou0129/Mumax3_spinresonator/cmd/mumax3`
    - `go install`
  * optional: install gnuplot if you want pretty graphs
    - Ubuntu: `sudo apt-get install gnuplot`

Your binary is now at `$GOPATH/bin/mumax3`

To do all at once on Ubuntu:
```
sudo apt-get install git golang-go gcc nvidia-cuda-toolkit nvidia-cuda-dev nvidia-340 gnuplot
GOPATH=$HOME go get -u -v github.com//jthou0129/Mumax3_spinresonator/cmd/mumax3
```


Example m-file for STO Maser Simulation
------------
sizeX := 1280e-9
sizeY := 1280e-9
sizeZ := 5e-9

Nx := 128
Ny := 128
 
setgridsize(Nx, Ny, 1)
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ)
setGeom(ellipse(sizeX, sizeY))

// set up free layer
Msat  = 550e3
Aex   = 9e-12
alpha = 0.04
Ku1 = 0.181e6
anisU = vector(0,0,1)
m     = uniform(0, 1, 0)
B_ext=vector(0,0.18,0)

// set up spacer layer parameters
lambda       = 1
Pol          = 0.5669
epsilonprime = 0

// set up fixed layer polarization
angle := 90
px := cos(angle * pi/180)
py := sin(angle * pi/180)
fixedlayer = vector(px, py, 0)

// send current
Jtot :=-0.2304           // total current in A
area := sizeX*sizeY*pi/4
jc   := Jtot / area       // current density in A/m2
J = vector(0, 0, jc)

SetSolver(7)
tempvalue:=300.0
Temp=tempvalue

//
Zr:=50.0
wr:=2*pi*5e9
Qfactor:=1000.0
Brfvalue:=15

Lvalue:=Zr/wr
Cvalue:=1/wr/Zr
Rvalue:=sqrt(Lvalue/Cvalue)/Qfactor

SetL(Lvalue)
SetC(Cvalue)
SetR(Rvalue)
SetBrf(Brfvalue)
SetTempRes(tempvalue)
InitialI(0.0)
InitialV(0.0)

// schedule output & run
autosave(m, 100e-9)
tableadd(Vout)
tableadd(Iout)
tableautosave(2.5e-12)
run(12000e-9)




Contributing
------------

Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
