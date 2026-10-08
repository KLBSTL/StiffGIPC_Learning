#include "../StiffGIPC/solver/line_search_acceptance.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
int checks = 0;
void require(bool condition, const char* property)
{
    ++checks;
    if(!condition) throw std::runtime_error(property);
}
void expect(gipc::LineSearchCheck check, gipc::LineSearchAction action, const char* reason)
{
    require(check.action == action && std::string(check.reason) == reason, reason);
}
}

int main()
{
    try
    {
        using Action = gipc::LineSearchAction;
        const double infinity = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double largest = std::numeric_limits<double>::max();
        const double tiny = std::numeric_limits<double>::denorm_min();

        // Independent acceptance boundaries, including the last permitted trial.
        for(int backtracks = 0; backtracks <= 9; ++backtracks)
        {
            for(double alpha : {1.0, 0.5, tiny})
            {
                auto equal = gipc::check_line_search_acceptance(10, 10, 0, alpha, backtracks);
                expect(equal, Action::accept, "accepted");
                require(equal.threshold == 10, "zero Armijo slope must require nonincrease");
                expect(gipc::check_line_search_acceptance(10, 9, 0, alpha, backtracks),
                       Action::accept, "accepted");
                expect(gipc::check_line_search_acceptance(10, std::nextafter(10.0, infinity),
                                                         0, alpha, backtracks),
                       backtracks == 9 ? Action::fail : Action::backtrack,
                       backtracks == 9 ? "energy_backtrack_budget_exhausted" : "energy_above_threshold");
            }
        }
        expect(gipc::check_line_search_acceptance(10, 9.5, -1, 0.5, 0), Action::accept, "accepted");
        expect(gipc::check_line_search_acceptance(10, std::nextafter(9.5, infinity), -1, 0.5, 0),
               Action::backtrack, "energy_above_threshold");
        for(double bad : {nan, infinity, -infinity})
        {
            expect(gipc::check_line_search_acceptance(bad, 0, 0, 1, 0), Action::fail, "nonfinite_baseline_energy");
            expect(gipc::check_line_search_acceptance(10, bad, 0, 1, 0), Action::fail, "nonfinite_trial_energy");
            expect(gipc::check_line_search_acceptance(10, 9, bad, 1, 0), Action::fail, "nonfinite_armijo_slope");
            expect(gipc::check_line_search_acceptance(10, 9, 0, bad, 0), Action::fail, "invalid_alpha");
        }
        for(double bad : {0.0, -1.0})
            expect(gipc::check_line_search_acceptance(10, 9, 0, bad, 0), Action::fail, "invalid_alpha");
        expect(gipc::check_line_search_acceptance(largest, 0, largest, 1, 0), Action::fail,
               "nonfinite_acceptance_threshold");
        expect(gipc::check_line_search_acceptance(0, 0, largest, 2, 0), Action::fail,
               "nonfinite_acceptance_threshold");
        for(int bad : {-1, 10})
            expect(gipc::check_line_search_acceptance(10, 9, 0, 1, bad), Action::fail, "invalid_backtrack_count");

        double next = 0;
        require(gipc::line_search_half_step(1, 1, next) && next == 0.5, "ordinary half step");
        require(gipc::line_search_half_step(1, 0.25, next) && next == 0.25, "CFL clamp");
        require(gipc::line_search_half_step(1, infinity, next) && next == 0.5, "unrestricted CFL");
        require(!gipc::line_search_half_step(tiny, 1, next), "underflow must fail before zero step");
        for(double bad : {nan, infinity, -infinity, 0.0, -1.0})
            require(!gipc::line_search_half_step(bad, 1, next), "invalid old step");
        for(double bad : {nan, -infinity, 0.0, -1.0})
            require(!gipc::line_search_half_step(1, bad, next), "invalid CFL bound");

        // A post-intersection state cannot inherit acceptance from an older trial.
        expect(gipc::check_line_search_acceptance(10, 9, 0, 0.5, 9), Action::accept, "accepted");
        expect(gipc::check_line_search_acceptance(10, 11, 0, 0.25, 9), Action::fail,
               "energy_backtrack_budget_exhausted");

        // Representable half steps terminate, so an intersection loop cannot
        // keep repeating an unchanged or zero alpha indefinitely.
        double alpha = 1;
        int halvings = 0;
        while(gipc::line_search_half_step(alpha, infinity, next))
        {
            require(next < alpha && next > 0, "strict progress at every half step");
            alpha = next;
            require(++halvings <= 1074, "finite representable half-step sequence");
        }
        require(alpha == tiny && halvings == 1074, "subnormal half-step endpoint");
        std::cout << "line_search_acceptance: " << checks << " checks passed\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "line_search_acceptance: " << error.what() << '\n';
        return 1;
    }
}
