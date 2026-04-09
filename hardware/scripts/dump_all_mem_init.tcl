# dump_all_mem_init.tcl
#
# Walk every RAMB cell under tinyissimoyolo_accel_0/inst/u_core/, group
# by parent memory instance, and print a per-group summary:
#   - count of cells in the group
#   - count with all-zero INIT_00
#   - first cell's INIT_00 (sample)
#
# This is a one-shot health check across all BRAM-backed memories
# (weight ROM, qp packed ROM, silu LUT, accumulator). If any group has
# zero non-zero cells, that's the smoking gun for "ROM init didn't make
# it into the routed dcp".
#
# Output: /tmp/all_mem_init.txt
#
# Usage (headless):
#   vivado -mode batch -nojournal -nolog -source hardware/scripts/dump_all_mem_init.tcl
#
# History: developed during the Apr 7-8 2026 sim-vs-silicon debug
# session as the broader BRAM init audit after dump_wt_mem_init.tcl
# verified the weight ROM was OK. Result: all 4 BRAM groups
# (u_wt_mem, u_qp_mem, u_silu_mem, u_inference_hdl/u_acc_mem) check out;
# bias/m0/nshift/zp_in/zp_out are not BRAM-backed and therefore not
# captured here.

set dcp_path "hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper_routed.dcp"
set out_path "/tmp/all_mem_init.txt"

puts "Opening checkpoint: $dcp_path"
open_checkpoint $dcp_path

set fp [open $out_path "w"]
puts $fp "=== all RAMB cells under tinyissimoyolo_accel ==="
puts $fp ""

set all_rams [get_cells -hierarchical -filter \
    {(REF_NAME == RAMB36E2 || REF_NAME == RAMB18E2) \
     && NAME =~ *tinyissimoyolo_accel_0/inst/u_core/*}]

set total [llength $all_rams]
puts "Found $total RAMB cells under tinyissimoyolo_accel_0/inst/u_core/"
puts $fp "total cells: $total"
puts $fp ""

# Group by the first 1-2 hierarchy components after u_core/
array set groups {}
array set group_zero {}
array set group_sample {}

foreach cell $all_rams {
    set name [get_property NAME $cell]
    set after_core ""
    if {[regexp {u_core/(.*)} $name -> after_core]} {
        set parts [split $after_core "/"]
        set group ""
        foreach p $parts {
            if {[regexp {^(ram_reg|RAMB|cascade|gen_)} $p]} { break }
            if {$group eq ""} { set group $p } else { set group "$group/$p" }
        }
    } else {
        set group "<unknown>"
    }

    set init_00 [get_property INIT_00 $cell]
    set is_zero 0
    set hex [regsub {^256'h} $init_00 ""]
    if {[regexp {^0+$} $hex]} { set is_zero 1 }

    if {[info exists groups($group)]} {
        incr groups($group)
    } else {
        set groups($group) 1
        set group_zero($group) 0
        set group_sample($group) $init_00
    }
    if {$is_zero} { incr group_zero($group) }
}

puts $fp [format "%-50s %8s %8s %s" "group" "cells" "all0" "sample INIT_00 (first 32 hex)"]
puts $fp [string repeat "-" 130]
foreach group [lsort [array names groups]] {
    set count $groups($group)
    set z $group_zero($group)
    set sample $group_sample($group)
    set sample_hex [regsub {^256'h} $sample ""]
    set sample_short [string range $sample_hex 0 31]
    puts $fp [format "%-50s %8d %8d %s" $group $count $z $sample_short]
}

close $fp
puts "Wrote $out_path"
