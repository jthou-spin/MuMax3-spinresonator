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

Please see STO1280nmBrf15Fr5B1800T300.mx3 in test folder for an example m-file for STO Maser simulation

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



Contributing
------------

Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
