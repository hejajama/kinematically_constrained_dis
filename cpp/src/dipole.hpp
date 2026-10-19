// Dipole S-matrix models.
//
// The quadrupole, the integrands and the cross sections are templates on the
// dipole model, which only has to provide
//
//   AtRapidity at(double Y) const;            // with
//   double AtRapidity::S(double r) const;     // S(r) at that rapidity
//
// at() is called once per integrand evaluation and S() for each of the
// dipoles at that point, so per-rapidity work belongs in at().
//
// BKDipole wraps a tabulated rcbk solution and an interpolator (see
// interpolation.hpp for how to add one). GBWDipole is an analytic model used
// to test the quadrupole independently of any interpolation.

#ifndef DIS_DIPOLE_HPP
#define DIS_DIPOLE_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "interpolation.hpp"
#include "rcbk_table.hpp"

namespace dis {

struct DipoleOptions {
    // Clip the interpolated N to [0, 1] (BKDipole clamp_N in Python).
    bool clip_N = true;
    // Lower bound on S = 1 - N, so that log(S) in the quadrupole is finite.
    double S_floor = 1e-14;
};

// BK-evolved dipole N(r, Y) from an rcbk file, as BKDipole in BK_dipole.py
// with its default settings: a query outside the tabulated grid is evaluated
// at the nearest boundary (r and Y clamped separately), N is clipped to
// [0, 1], and S = max(1 - N, S_floor).
template <class Interpolator>
class BKDipole {
public:
    explicit BKDipole(RcbkTable table, DipoleOptions options = {})
        : table_(std::move(table)), interp_(table_), options_(options) {}

    explicit BKDipole(const std::string &filename, DipoleOptions options = {})
        : BKDipole(RcbkTable::read(filename), options) {}

    class AtRapidity {
    public:
        double N(double r) const {
            double value = slice_(std::clamp(r, r_min_, r_max_));
            if (clip_N_) value = std::clamp(value, 0.0, 1.0);
            return value;
        }
        double S(double r) const { return std::max(1.0 - N(r), S_floor_); }

    private:
        friend class BKDipole;
        typename Interpolator::Slice slice_;
        double r_min_, r_max_, S_floor_;
        bool clip_N_;
    };

    AtRapidity at(double Y) const {
        AtRapidity a;
        a.slice_ = interp_.at(std::clamp(Y, table_.Y_min(), table_.Y_max()));
        a.r_min_ = table_.r_min();
        a.r_max_ = table_.r_max();
        a.S_floor_ = options_.S_floor;
        a.clip_N_ = options_.clip_N;
        return a;
    }

    double N(double r, double Y) const { return at(Y).N(r); }
    double S(double r, double Y) const { return at(Y).S(r); }

    const RcbkTable &table() const { return table_; }
    const Interpolator &interpolator() const { return interp_; }

private:
    RcbkTable table_;
    Interpolator interp_;
    DipoleOptions options_;
};

// GBW dipole S(r) = exp(-r^2 Qs^2 / 4), independent of Y.
class GBWDipole {
public:
    explicit GBWDipole(double Qs2) : Qs2_(Qs2) {}

    class AtRapidity {
    public:
        double S(double r) const { return std::exp(-r * r * Qs2_ / 4); }

    private:
        friend class GBWDipole;
        double Qs2_ = 0;
    };

    AtRapidity at(double /*Y*/) const {
        AtRapidity a;
        a.Qs2_ = Qs2_;
        return a;
    }

private:
    double Qs2_;
};

}  // namespace dis

#endif  // DIS_DIPOLE_HPP
