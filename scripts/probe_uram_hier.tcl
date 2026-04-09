# Probe the URAM hierarchy for each sim flavor — verify whether
# dut.u_fmap_a.ram exists as a hierarchical handle, since the post-synth
# / post-impl netlists replace sdp_ram with URAM288 primitives that
# don't expose a `reg ram[...]`.

set flow [lindex $argv 0]
if {$flow eq ""} { set flow "behav" }

open_project hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr

switch -- $flow {
    behav      { launch_simulation -mode behavioral -simset sim_1 }
    post_synth { launch_simulation -mode post-synthesis -type functional -simset sim_1 }
    post_impl  { launch_simulation -mode post-implementation -type functional -simset sim_1 }
    post_timing { launch_simulation -mode post-implementation -type timing -simset sim_1 }
}

# Run a tiny bit to settle the elaboration
run 100ns

puts "=== probing tb_tinyissimoyolo_accel/dut hierarchy ==="
puts "  flow: $flow"

# Try several hierarchy paths
foreach path {
    /tb_tinyissimoyolo_accel/dut/u_fmap_a/ram
    /tb_tinyissimoyolo_accel/dut/u_fmap_a
    /tb_tinyissimoyolo_accel/dut
    /tb_tinyissimoyolo_accel/dut/u_inference_hdl/layer_idx
    /tb_tinyissimoyolo_accel/dut/u_inference_hdl
} {
    if {[catch {set objs [get_objects -r $path]} err]} {
        puts "  $path -> ERROR: $err"
    } else {
        puts "  $path -> [llength $objs] objects"
        if {[llength $objs] > 0 && [llength $objs] <= 5} {
            foreach o $objs { puts "    $o" }
        }
    }
}

# Try get_scopes
puts ""
puts "=== scopes under /tb_tinyissimoyolo_accel/dut ==="
if {[catch {set scopes [get_scopes -r /tb_tinyissimoyolo_accel/dut/*]} err]} {
    puts "  ERROR: $err"
} else {
    foreach s $scopes { puts "  $s" }
}

close_sim -quiet
exit
