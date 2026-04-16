# Headless sim runner for tb_tinyissimoyolo_accel.
#
# Usage:  vivado -mode batch -source scripts/run_sim_headless.tcl -tclargs <flow>
#   where <flow> is one of:
#       behav        — behavioral simulation
#       post_synth   — post-synthesis functional
#       post_impl    — post-implementation functional
#       post_timing  — post-implementation timing (with SDF)

set flow [lindex $argv 0]
if {$flow eq ""} { set flow "behav" }

puts "=========================================="
puts " run_sim_headless.tcl  flow=$flow"
puts "=========================================="

# Auto-recreate the project if .xpr is missing.
if {![file exists hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr]} {
    puts "==> XPR not found — recreating project..."
    source scripts/recreate_project.tcl
}

open_project hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr

# Pick up any new testbench files dropped under hardware/testbench/ since
# the project was last recreated. recreate_project.tcl globs the same
# directory; this keeps incremental sim runs in sync without a full rebuild.
set tb_dir [file normalize hardware/testbench]
set existing_tb [lsort [get_files -of [get_filesets sim_1] -filter {FILE_TYPE == SystemVerilog} -quiet]]
foreach f [glob -nocomplain $tb_dir/*.sv] {
    if {[lsearch $existing_tb $f] < 0} {
        puts "==> adding new TB source to sim_1: [file tail $f]"
        add_files -norecurse -fileset sim_1 $f
    }
}
update_compile_order -fileset sim_1 -quiet

# Make sure the sim top is what we expect
set top [get_property top [get_filesets sim_1]]
puts "=> sim_1 top: $top"

switch -- $flow {
    behav {
        launch_simulation -mode behavioral -simset sim_1
    }
    post_synth {
        launch_simulation -mode post-synthesis -type functional -simset sim_1
    }
    post_impl {
        launch_simulation -mode post-implementation -type functional -simset sim_1
    }
    post_timing {
        launch_simulation -mode post-implementation -type timing -simset sim_1
    }
    default {
        puts "ERROR: unknown flow '$flow'"
        exit 1
    }
}

# launch_simulation doesn't auto-run to $finish in batch mode unless we
# explicitly run. The TB calls $finish so this terminates cleanly.
puts "=> running simulation (will halt at \$finish)"
run all

puts "=> closing simulation"
close_sim -quiet

# If cycle_monitor produced a CSV, surface its location
set csv_candidates [glob -nocomplain hardware/vivado/tinyissimoyolo/tinyissimoyolo.sim/sim_1/*/xsim/cycle_breakdown.csv]
foreach csv $csv_candidates {
    puts "=> cycle breakdown CSV: [file normalize $csv]"
}

puts "=> done"
exit
