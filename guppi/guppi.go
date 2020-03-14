package main

import (
    "fmt"
	"os"
	"strings"
	"strconv"
	"encoding/json"
	"io/ioutil"
)

type Head map[string]string
type Heads map[int]Head

func toInt64(s string) int64 {
	i, _ := strconv.Atoi(s)
	return int64(i)
}

func toFloat64(s string) float64 {
	i, _ := strconv.ParseFloat(s, 64)
	return i
}

func exportHeads(h Heads) {
	jsonString, _ := json.Marshal(h)
	ioutil.WriteFile("heads.json", jsonString, 0755)
}

func main() {
	pos := int64(0)
	headsCount := 0
	buff := make([]byte, 80)
	heads := make(Heads)

	r := strings.NewReplacer("'", "", " ", "")

	f, err := os.Open("/media/luigifcruz/HDD1/SETI/blc6_guppi_57388_HIP113357_0010.0000.raw")
	defer f.Close()
	if err != nil {
        panic(err)
	}
	
	for {
		_, err := f.Seek(pos, 0)
		n, err := f.Read(buff)
		pos += int64(n)

		if err != nil {
			break
		}

		s := strings.Split(string(buff), "= ")
		if strings.Contains(s[0], "END      ") {
			pos += toInt64(heads[headsCount]["BLOCSIZE"])
			heads[headsCount]["FRSTPINT"] = strconv.Itoa(int(pos))
			headsCount += 1
			continue
		}

		if heads[headsCount] == nil {
			heads[headsCount] = make(Head)
		}
		
		heads[headsCount][r.Replace(s[0])] = r.Replace(s[1])
	}

	OBSNCHAN := toInt64(heads[0]["OBSNCHAN"])
	NPOL := toInt64(heads[0]["NPOL"])
	NBITS := toInt64(heads[0]["NBITS"])
	DIRECTIO := toInt64(heads[0]["DIRECTIO"])
	BLOCSIZE := toInt64(heads[0]["BLOCSIZE"])
	OBSBW := toFloat64(heads[0]["OBSBW"])

	if NPOL > 2 {
		fmt.Printf("NPOL: Correcting anomalous polarization number (%d)\n", NPOL)
		NPOL = 2
	}

	NTIME := (BLOCSIZE * 8) / (2 * NPOL * OBSNCHAN * NBITS)
	CHSIZE := NPOL * (NBITS / 4) * NTIME

	fmt.Printf("OBSNCHAN:	%d\n", OBSNCHAN)
	fmt.Printf("NPOL:		%d\n", NPOL)
	fmt.Printf("NBITS:		%d\n", NBITS)
	fmt.Printf("DIRECTIO:	%d\n", DIRECTIO)
	fmt.Printf("BLOCSIZE:	%d\n", BLOCSIZE)
	fmt.Printf("NTIME:		%d\n", NTIME)
	fmt.Printf("CHSIZE:		%d\n", CHSIZE)
	fmt.Printf("OBSBW:		%f\n", OBSBW)
	fmt.Printf("HEADS:		%d\n", len(heads))

	exportHeads(heads)

	if NBITS != 8 {
		fmt.Printf("NBITS: Not supported (%d)\n", NBITS)
		os.Exit(1)
	}

	if DIRECTIO != 0 {
		fmt.Printf("DIRECTIO: Not supported (%d)\n", DIRECTIO)
		os.Exit(1)
	}

	buff = make([]byte, BLOCSIZE)
	chFiles := make(map[int64]*os.File)

	for i := int64(0); i < OBSNCHAN; i++ {
		fileName := fmt.Sprintf("data/HIP113357/FILE_C%d", i)
		chFiles[i], err = os.Create(fileName)
		defer chFiles[i].Close()
		if err != nil {
			panic(err)
		}
	}

	for _, head := range heads {
		FRSTPINT := toInt64(head["FRSTPINT"])

		_, err := f.Seek(FRSTPINT, 0)
		_, err = f.Read(buff)

		for i := int64(0); i < OBSNCHAN; i++ {
			off := CHSIZE * i
			_, err = chFiles[i].Write(buff[off:off+CHSIZE])
			if err != nil {
				panic(err)
			}
		}
	}
}