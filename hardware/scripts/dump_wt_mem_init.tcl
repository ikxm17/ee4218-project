# dump_wt_mem_init.tcl
#
# Inspect the routed checkpoint and dump INIT_xx properties for every
# RAMB36E2 cell that maps from inference_top.u_wt_mem (the weight ROM
# that was declared as URAM but downgraded to BRAM by Vivado because
# URAMs cannot load from $readmemh files at synthesis time).
#
# Output: /tmp/wt_mem_init.txt
#   - Cell count summary
#   - For each cell: full hierarchical name, LOC, RAM_MODE, INIT_00,
#     INITP_00 (just the first init slot — enough to check whether
#     each cell contains weight data or is all-zero)
#
# Usage (headless):
#   vivado -mode batch -nojournal -nolog -source hardware/scripts/dump_wt_mem_init.tcl
#
# History: developed during the Apr 7-8 2026 sim-vs-silicon debug
# session to verify whether weight ROM corruption was the cause of the
# silicon mismatch. Result: cell[0]'s INIT_00 matched weight_rom.mem
# byte-for-byte, ruling out BRAM init corruption as the cause. The
# 25 cells with all-zero INIT_00 are upper-bits BRAMs holding the
# zero high bits of weight_rom.mem's 128-bit words.

set dcp_path "hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper_routed.dcp"
set out_path "/tmp/wt_mem_init.txt"

puts "Opening checkpoint: $dcp_path"
open_checkpoint $dcp_path

set fp [open $out_path "w"]
puts $fp "=== weight_rom BRAM init dump ==="
puts $fp "checkpoint: $dcp_path"
puts $fp ""

set wt_cells [get_cells -hierarchical -filter \
    {(REF_NAME == RAMB36E2 || REF_NAME == RAMB18E2) && NAME =~ *u_wt_mem*}]

set n [llength $wt_cells]
puts "Found $n RAMB cells matching *u_wt_mem*"
puts $fp "cell count: $n"
puts $fp ""

if {$n == 0} {
    puts $fp "ERROR: no cells matched *u_wt_mem*. Trying broader search..."
    set wt_cells [get_cells -hierarchical -filter {REF_NAME == RAMB36E2 || REF_NAME == RAMB18E2}]
    puts $fp "all RAMB cells in design: [llength $wt_cells]"
    foreach c [lrange $wt_cells 0 9] {
        puts $fp "  $c"
    }
}

set i 0
set zero_count 0
foreach cell $wt_cells {
    set name [get_property NAME $cell]
    set ref [get_property REF_NAME $cell]
    set loc [get_property LOC $cell]
    set ram_mode ""
    catch {set ram_mode [get_property RAM_MODE $cell]}
    set read_width_a ""
    catch {set read_width_a [get_property READ_WIDTH_A $cell]}
    set read_width_b ""
    catch {set read_width_b [get_property READ_WIDTH_B $cell]}
    set init_00 [get_property INIT_00 $cell]
    set initp_00 ""
    catch {set initp_00 [get_property INITP_00 $cell]}

    set init_00_hex [regsub {^256'h} $init_00 ""]
    if {$init_00_hex == ""} { set init_00_hex $init_00 }
    if {[string match {*0000000000000000000000000000000000000000000000000000000000000000*} $init_00_hex]} {
        incr zero_count
    }

    puts $fp "cell\[$i\] = $name"
    puts $fp "  ref:        $ref"
    puts $fp "  loc:        $loc"
    puts $fp "  ram_mode:   $ram_mode"
    puts $fp "  rd_width_a: $read_width_a   rd_width_b: $read_width_b"
    puts $fp "  INIT_00:    $init_00"
    puts $fp "  INITP_00:   $initp_00"
    puts $fp ""
    incr i
}

puts $fp ""
puts $fp "summary: $n cells total, $zero_count have all-zero INIT_00"
close $fp
puts "Wrote $out_path  ($n cells, $zero_count all-zero INIT_00)"
