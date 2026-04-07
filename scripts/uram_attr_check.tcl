# URAM primitive attribute inspection for u_fmap_a / u_fmap_b
#
# Reads OREG_*, REG_CAS_*, IREG_PRE_*, OREG_ECC_*, CASCADE_ORDER_*
# from every URAM288 instance under the u_fmap_a / u_fmap_b feature-map
# RAMs in the routed checkpoint, and reports the deterministic
# silicon read latency (per UG573 Table 37) per cell.
#
# Latency formula per UG573:
#   latency_silicon = IREG_PRE_B
#                   + (REG_CAS_B accumulating across cascade hops)
#                   + OREG_B
#                   + OREG_ECC_B
#
# Compare against the sdp_ram.sv template's modeled latency of 1 cycle
# (`dout_b <= ram[addr_b]`).

set dcp [lindex $argv 0]
if {$dcp eq ""} {
    set dcp "hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper_routed.dcp"
}

puts "=== opening checkpoint: $dcp ==="
open_checkpoint $dcp

puts "\n=== finding URAM288 cells under u_fmap_a / u_fmap_b ==="

# Try a couple of name patterns; the hierarchy may be flattened differently
# depending on whether the IP is OOC or in-context.
set pattern_list {
    {*u_fmap_a*}
    {*u_fmap_b*}
    {*fmap_a*}
    {*fmap_b*}
}

set fmap_urams [list]
foreach pat $pattern_list {
    set hits [get_cells -hierarchical -filter "REF_NAME == URAM288 && NAME =~ $pat" -quiet]
    if {[llength $hits] > 0} {
        puts "  pattern $pat -> [llength $hits] URAM288 cells"
        foreach h $hits {
            if {[lsearch -exact $fmap_urams $h] < 0} {
                lappend fmap_urams $h
            }
        }
    }
}

if {[llength $fmap_urams] == 0} {
    puts "WARNING: no fmap-prefixed URAM288 cells found by name. Listing ALL URAM288 cells in design instead:"
    set fmap_urams [get_cells -hierarchical -filter {REF_NAME == URAM288} -quiet]
    puts "  found [llength $fmap_urams] URAM288 cells in total"
}

puts "\n=== URAM attribute dump ==="
set total_count 0
foreach c $fmap_urams {
    incr total_count
    puts ""
    puts "  cell\[$total_count\]: $c"
    foreach a {OREG_A OREG_B OREG_ECC_A OREG_ECC_B \
               REG_CAS_A REG_CAS_B \
               IREG_PRE_A IREG_PRE_B \
               CASCADE_ORDER_A CASCADE_ORDER_B \
               BWE_MODE_A BWE_MODE_B \
               EN_AUTO_SLEEP_MODE \
               IS_RDB_REGCE_INVERTED} {
        if {[catch {get_property $a $c} val]} {
            puts "    $a = <not present>"
        } else {
            puts "    $a = $val"
        }
    }
}

puts "\n=== summary: silicon read latency per cell (port B) ==="
puts "  formula: IREG_PRE_B + OREG_B + OREG_ECC_B  (cascade contribution shown separately)"
puts "  --------"
foreach c $fmap_urams {
    set ireg [get_property IREG_PRE_B $c]
    set oreg [get_property OREG_B $c]
    set oecc [get_property OREG_ECC_B $c]
    set rcas [get_property REG_CAS_B $c]
    set ord  [get_property CASCADE_ORDER_B $c]

    set ireg_cyc [expr {$ireg eq "TRUE" ? 1 : 0}]
    set oreg_cyc [expr {$oreg eq "TRUE" ? 1 : 0}]
    set oecc_cyc [expr {$oecc eq "TRUE" ? 1 : 0}]
    set rcas_cyc [expr {$rcas eq "TRUE" ? 1 : 0}]

    set base_lat [expr {$ireg_cyc + $oreg_cyc + $oecc_cyc}]
    puts "  $c"
    puts "    IREG_PRE_B=$ireg OREG_B=$oreg OREG_ECC_B=$oecc REG_CAS_B=$rcas CASCADE_ORDER_B=$ord"
    puts "    base_latency_to_chain_exit (ignoring cascade hops) = $base_lat cycles"
    puts "    cascade_pipe_per_hop = $rcas_cyc cycles (REG_CAS_B contribution)"
}

puts "\n=== bonus: also dump URAM288 utilization for sanity ==="
report_utilization -no_pblock -file /dev/null -quiet
set urams [get_cells -hierarchical -filter {REF_NAME == URAM288} -quiet]
puts "  total URAM288 instances in routed design: [llength $urams]"

puts "\n=== done ==="
exit
