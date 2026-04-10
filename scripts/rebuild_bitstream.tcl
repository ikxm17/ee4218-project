# Headless rebuild of the playground bitstream after RTL edits.
#
# Order matters because the design has an OOC IP cache hazard:
#   1. Reset and regenerate the IP's output products (so the OOC synth
#      picks up the latest ip_repo/src/ files instead of the stale
#      ipshared/<hash>/src/ cache).
#   2. Reset top-level synth_1 and impl_1.
#   3. Launch OOC IP synth, wait.
#   4. Launch impl_1 (which auto-launches synth_1 first), wait.
#   5. Write bitstream as part of impl_1's to_step.
#   6. Export hardware platform (.xsa) for the deploy script.
#
# Per feedback_vivado_jobs.md, cap parallelism at -jobs 8 to avoid
# exhausting machine memory.

puts "=== rebuild_bitstream.tcl ==="

# Auto-recreate the project if .xpr is missing (e.g. after git clone
# or after git rm --cached removed it from tracking).
if {![file exists hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr]} {
    puts "==> XPR not found — recreating project..."
    source scripts/recreate_project.tcl
}

open_project hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr

# Identify runs. The OOC IP synth run name varies between project versions
# (sometimes playground_tinyissimoyolo_accel_0_0_synth_1, sometimes absent
# entirely). Auto-discover any *_synth_* run that isn't synth_1.
set top_runs [list synth_1 impl_1]
set ooc_runs [list]
foreach r [get_runs] {
    if {[string match "*_synth_*" $r] && $r ne "synth_1"} {
        lappend ooc_runs $r
    }
}
puts "  discovered OOC runs: $ooc_runs"

# Step 1: refresh BD output products. The accelerator IP is nested
# inside playground.bd, so we have to regenerate the parent BD to
# refresh the IP's source files (Vivado error 12-3563 if you try
# to generate the nested IP directly).
puts "\n=== refreshing BD output products (playground.bd) ==="
set bd_file [get_files playground.bd]
puts "  BD file: $bd_file"
reset_target {synthesis simulation} $bd_file -quiet
generate_target {synthesis simulation} $bd_file

# Step 2: reset runs
puts "\n=== resetting runs ==="
foreach r [concat $top_runs $ooc_runs] {
    if {[get_property STATUS [get_runs $r]] ne "Not started"} {
        puts "  reset_run $r"
        reset_run $r
    } else {
        puts "  $r already at 'Not started'"
    }
}

# Step 3: launch OOC IP synth first
puts "\n=== launch_runs OOC IP synth ==="
foreach r $ooc_runs {
    launch_runs $r -jobs 8
}
foreach r $ooc_runs {
    puts "  wait_on_run $r"
    wait_on_run $r
    set st [get_property STATUS [get_runs $r]]
    puts "  $r status: $st"
    if {[string match "*ERROR*" $st]} {
        puts "ERROR: OOC run $r did not complete successfully"
        exit 1
    }
}

# Step 4: launch top impl_1 (auto-launches synth_1 first)
#
# Hold-fix insurance: -tns_cleanup runs an extra hold-fix pass at the end
# of route_design that inserts LUT1 buffers (or reroutes) on any path with
# negative or near-zero hold slack. This is essential for the conv3d
# ACC_write_address -> ACC_write_address_d cone-A delay path which has
# only ~19 ps of hold margin in the default flow. Without -tns_cleanup,
# small placement perturbations (e.g. from new RTL features like HLS
# integration) can re-trigger the silicon-only +1 URAM shift bug.
puts "\n=== set route_design -tns_cleanup for hold-fix insurance ==="
set_property -name {STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS} -value {-tns_cleanup} -objects [get_runs impl_1]
puts "\n=== launch_runs impl_1 -to_step write_bitstream ==="
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
set st [get_property STATUS [get_runs impl_1]]
puts "  impl_1 status: $st"
if {[string match "*ERROR*" $st]} {
    puts "ERROR: impl_1 did not complete successfully"
    exit 1
}

# Step 5: export hardware platform
puts "\n=== writing hardware platform xsa ==="
write_hw_platform -fixed -include_bit -force hardware/output/playground.xsa

# Step 6: copy bit and hwh to hardware/output for the deploy script
puts "\n=== copying .bit and .hwh ==="
file copy -force hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper.bit hardware/output/playground.bit
file copy -force hardware/vivado/tinyissimoyolo/tinyissimoyolo.gen/sources_1/bd/playground/hw_handoff/playground.hwh hardware/output/playground.hwh

puts "\n=== done ==="
exit
