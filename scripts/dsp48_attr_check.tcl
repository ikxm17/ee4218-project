# DSP48E2 primitive attribute inspection for the convolver xbip_dsp48_macro
# instances in the routed checkpoint.
#
# Reads register-stage attributes (AREG/BREG/MREG/PREG/ADREG/...) and reports
# the deterministic silicon pipeline depth so we can compare against the
# 'fill_count' assumption baked into hardware/rtl/convolver.v.
#
# Pipeline depth (per UG579):
#   typical mult-add chain feeding A/B -> M -> P:
#       depth = max(AREG, BREG) + MREG + PREG
#   xbip_dsp48_macro defaults usually fall in {3, 4} cycles.

set dcp [lindex $argv 0]
if {$dcp eq ""} {
    set dcp "hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper_routed.dcp"
}

puts "=== opening checkpoint: $dcp ==="
open_checkpoint $dcp

puts "\n=== finding all DSP48E2 cells ==="
set dsps [get_cells -hierarchical -filter {REF_NAME == DSP48E2} -quiet]
puts "  total DSP48E2 cells in routed design: [llength $dsps]"

# Group by parent hierarchy: anything that mentions conv / xbip / dsp48
set conv_dsps [list]
foreach c $dsps {
    if {[string match "*conv*" $c] || [string match "*xbip*" $c] || [string match "*dsp48*" $c]} {
        lappend conv_dsps $c
    }
}
puts "  conv-related DSP48E2 cells: [llength $conv_dsps]"

if {[llength $conv_dsps] == 0} {
    puts "WARNING: no conv-named DSP48E2 cells found. Falling back to all DSP48E2."
    set conv_dsps $dsps
}

puts "\n=== first 10 conv-related DSP48E2 cells: full attribute dump ==="
set i 0
foreach c [lrange $conv_dsps 0 9] {
    incr i
    puts ""
    puts "  cell\[$i\]: $c"
    foreach a {AREG BREG CREG DREG MREG PREG ADREG ACASCREG BCASCREG \
               INMODEREG OPMODEREG ALUMODEREG CARRYINREG CARRYINSELREG \
               USE_MULT A_INPUT B_INPUT USE_SIMD} {
        if {[catch {get_property $a $c} val]} {
            puts "    $a = <not present>"
        } else {
            puts "    $a = $val"
        }
    }
}

puts "\n=== distinct register tuples across all conv DSPs ==="
set tuples [dict create]
foreach c $conv_dsps {
    set t "A=[get_property AREG $c] B=[get_property BREG $c] M=[get_property MREG $c] P=[get_property PREG $c] AD=[get_property ADREG $c] C=[get_property CREG $c] D=[get_property DREG $c]"
    dict incr tuples $t
}
foreach t [dict keys $tuples] {
    puts "  $t  count=[dict get $tuples $t]"
}

puts "\n=== calculated pipeline depth per tuple ==="
puts "  depth = max(AREG,BREG) + MREG + PREG  (typical mult chain)"
foreach t [dict keys $tuples] {
    # parse "A=x B=x M=x P=x ..."
    if {[regexp {A=(\d+) B=(\d+) M=(\d+) P=(\d+)} $t -> areg breg mreg preg]} {
        set in_max [expr {$areg > $breg ? $areg : $breg}]
        set depth [expr {$in_max + $mreg + $preg}]
        puts "  $t  -> depth=$depth"
    } else {
        puts "  $t  -> (parse failed)"
    }
}

puts "\n=== distinct hierarchy parents of conv DSPs (first level above DSP) ==="
set parents [dict create]
foreach c $conv_dsps {
    set p [file dirname $c]
    set pp [file dirname $p]
    dict incr parents $pp
}
foreach p [dict keys $parents] {
    puts "  $p  count=[dict get $parents $p]"
}

puts "\n=== done ==="
close_design
exit
