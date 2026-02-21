//go:build windows

package sensors

import (
	"syscall"
	"unsafe"
)

type WindowsSampler struct {
	prevIdle   uint64
	prevKernel uint64
	prevUser   uint64
}

type filetime struct {
	dwLowDateTime  uint32
	dwHighDateTime uint32
}

func NewSampler() Sampler { return &WindowsSampler{} }

func (s *WindowsSampler) Sample(path string) (Snapshot, error) {
	var out Snapshot
	p, _ := syscall.UTF16PtrFromString(path)
	var freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes uint64
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GetDiskFreeSpaceExW")
	r1, _, _ := proc.Call(uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(&freeBytesAvailable)), uintptr(unsafe.Pointer(&totalNumberOfBytes)), uintptr(unsafe.Pointer(&totalNumberOfFreeBytes)))
	if r1 != 0 {
		out.DiskFreeBytes = totalNumberOfFreeBytes
		out.DiskTotalBytes = totalNumberOfBytes
	}
	type memStatusEx struct {
		dwLength                uint32
		dwMemoryLoad            uint32
		ullTotalPhys            uint64
		ullAvailPhys            uint64
		ullTotalPageFile        uint64
		ullAvailPageFile        uint64
		ullTotalVirtual         uint64
		ullAvailVirtual         uint64
		ullAvailExtendedVirtual uint64
	}
	var mse memStatusEx
	mse.dwLength = uint32(unsafe.Sizeof(mse))
	procMem := k32.NewProc("GlobalMemoryStatusEx")
	r2, _, _ := procMem.Call(uintptr(unsafe.Pointer(&mse)))
	if r2 != 0 {
		out.RamTotalBytes = mse.ullTotalPhys
		out.RamFreeBytes = mse.ullAvailPhys
	}
	var idle, kernel, user filetime
	procTimes := k32.NewProc("GetSystemTimes")
	r3, _, _ := procTimes.Call(uintptr(unsafe.Pointer(&idle)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user)))
	if r3 != 0 {
		id := ftToU64(idle)
		ke := ftToU64(kernel)
		us := ftToU64(user)
		if s.prevKernel > 0 {
			dIdle := id - s.prevIdle
			dKernel := ke - s.prevKernel
			dUser := us - s.prevUser
			dTotal := dKernel + dUser
			if dTotal > 0 {
				util := 1.0 - float64(dIdle)/float64(dTotal)
				if util < 0 {
					util = 0
				}
				if util > 1 {
					util = 1
				}
				out.CPUUtil = util
			}
		}
		s.prevIdle, s.prevKernel, s.prevUser = id, ke, us
	}
	return out, nil
}

func ftToU64(ft filetime) uint64 { return uint64(ft.dwHighDateTime)<<32 + uint64(ft.dwLowDateTime) }
