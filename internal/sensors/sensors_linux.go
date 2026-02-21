//go:build linux

package sensors

import (
	"bufio"
	"errors"
	"os"
	"strconv"
	"strings"
	"syscall"
)

type LinuxSampler struct {
	prevTotal uint64
	prevIdle  uint64
}

func NewSampler() Sampler { return &LinuxSampler{} }

func (s *LinuxSampler) Sample(path string) (Snapshot, error) {
	var out Snapshot
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err == nil {
		out.DiskFreeBytes = st.Bavail * uint64(st.Bsize)
		out.DiskTotalBytes = st.Blocks * uint64(st.Bsize)
	}
	meminfo, err := os.Open("/proc/meminfo")
	if err != nil {
		return out, err
	}
	defer meminfo.Close()
	var memAvailKB, memTotalKB uint64
	sc := bufio.NewScanner(meminfo)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			memTotalKB = parseKB(line)
		}
		if strings.HasPrefix(line, "MemAvailable:") {
			memAvailKB = parseKB(line)
		}
	}
	if memTotalKB > 0 {
		out.RamTotalBytes = memTotalKB * 1024
		out.RamFreeBytes = memAvailKB * 1024
	}
	total, idle, err := readProcStat()
	if err == nil && total > 0 {
		if s.prevTotal > 0 && total > s.prevTotal {
			dt := total - s.prevTotal
			di := idle - s.prevIdle
			if dt > 0 {
				util := 1.0 - float64(di)/float64(dt)
				if util < 0 {
					util = 0
				}
				if util > 1 {
					util = 1
				}
				out.CPUUtil = util
			}
		}
		s.prevTotal = total
		s.prevIdle = idle
	}
	return out, nil
}

func parseKB(line string) uint64 {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return 0
	}
	v, _ := strconv.ParseUint(fields[1], 10, 64)
	return v
}

func readProcStat() (total uint64, idle uint64, err error) {
	f, e := os.Open("/proc/stat")
	if e != nil {
		return 0, 0, e
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	if !sc.Scan() {
		return 0, 0, errors.New("empty /proc/stat")
	}
	line := sc.Text()
	if !strings.HasPrefix(line, "cpu ") {
		return 0, 0, errors.New("no cpu line")
	}
	fields := strings.Fields(line)
	var vals []uint64
	for i := 1; i < len(fields); i++ {
		v, _ := strconv.ParseUint(fields[i], 10, 64)
		vals = append(vals, v)
	}
	for _, v := range vals {
		total += v
	}
	if len(vals) >= 4 {
		idle = vals[3]
	}
	if len(vals) >= 5 {
		idle += vals[4]
	}
	return total, idle, nil
}
