
// Derived from https://github.com/gonum/gonum/blob/master/stat/distuv/binomial.go

// Copyright (c) 2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package input

import (
    "math"
    "math/rand"
)

func Binomial(bn float64, bp float64) float64 {
    // NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING (ISBN 0-521-43108-5)
    // p. 295-6
    // http://www.aip.de/groups/soe/local/numres/bookcpdf/c7-3.pdf

    p := bp
    if p > 0.5 {
        p = 1.0 - p
    }
    am := bn * p

    if bn < 25 {
        // Use direct method.
        bnl := 0.0
        for i := 0; i < int(bn); i++ {
            if rand.Float64() < p {
                bnl++
            }
        }
        if p != bp {
            return bn - bnl
        }
        return bnl
    }

    if am < 1 {
        // Use rejection method with Poisson proposal.
        // constant for rejection sampling (https://en.wikipedia.org/wiki/Rejection_sampling)
        const logM = 2.6e-2
        var bnl float64
        z := -p
        pclog := (1 + 0.5 * z) * z / (1 + (1 + 1.0 / 6 * z) * z) // Pade' approximant of log(1 + x)
        for {
            bnl = 0.0
            t := 0.0
            for i := 0; i < int(bn); i++ {
                t += rand.ExpFloat64()
                if t >= am {
                    break
                }
                bnl++
            }
            bnlc := bn - bnl
            z = -bnl / bn
            log1p := (1 + 0.5 * z) * z / (1 + (1 + 1.0 / 6 * z) * z)
            // uses Stirling's expansion of log(n!)
            t = (bnlc + 0.5) * log1p + bnl - bnlc * pclog + 1 / (12 * bnlc) - am + logM 
            if rand.ExpFloat64() >= t {
                break
            }
        }
        if p != bp {
            return bn - bnl
        }
        return bnl
    }

    // Original algorithm samples from a Poisson distribution with the
    // appropriate expected value. However, the Poisson approximation is
    // asymptotic such that the absolute deviation in probability is O(1/n).
    // Rejection sampling produces exact variates with at worst less than 3%
    // rejection with miminal additional computation.

    // Use rejection method with Cauchy proposal.
    g, _ := math.Lgamma(bn+1)
    plog := math.Log(p)
    pclog := math.Log1p(-p)
    sq := math.Sqrt(2*am*(1-p))
    for {
        var em, y float64
        for {
            y = math.Tan(math.Pi*rand.Float64())
            em = sq * y + am
            if em >= 0 && em < bn + 1 {
                break
            }
        }
        em = math.Floor(em)
        lg1, _ := math.Lgamma(em+1)
        lg2, _ := math.Lgamma(bn-em+1)
        t := 1.2 * sq * (1 + y * y) * math.Exp(g-lg1-lg2+em*plog+(bn-em)*pclog)
        if rand.Float64() <= t {
            if p != bp {
                return bn - em
            }
            return em
        }
    }
}

