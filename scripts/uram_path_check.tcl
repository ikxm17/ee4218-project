# URAM read-path sequential inspection for u_fmap_a / u_fmap_b
#
# Hypothesis: Vivado synth/impl may have inserted a hidden fabric flop
# on the URAM288 port-B address path (pixel_bram_addr -> ADDR_B[*]) or
# on the data path (DOUT_B[*] -> pixel_bram_data) even though the source
# RTL shows a single-cycle sdp_ram read.
#
# This script walks every URAM288 cell under u_fmap_a and u_fmap_b and:
#   1. Lists the immediate driver of every ADDR_B pin. If the driver is
#      an FD* sequential cell, we flag it.
#   2. Lists the immediate load of every DOUT_B pin. If the load is an
#      FD* sequential cell (data capture flop), we flag it.
#   3. Reports the worst-case combinational timing path from any flop
#      output to the URAM ADDR_B inputs and from URAM DOUT_B outputs
#      to any downstream flop D pin.
#
# A pure 1-cycle read must have:
#   - ADDR_B driven combinationally from conv3d's pixel_bram_addr flop
#     (so the driving cell may be a flop, but there must NOT be an
#     intermediate flop between that flop and ADDR_B)
#   - DOUT_B fanning out combinationally to consumers (conv3d's act_zp
#     mux), with no intermediate capture flop before the mux.
#
# Any extra flop in between means the effective silicon latency is
# 2 cycles while source RTL / post-impl behavioural sim model it as 1.

set dcp [lindex $argv 0]
if {$dcp eq ""} {
    set dcp "hardware/vivado/tinyissimoyolo/tinyissimoyolo.runs/impl_1/playground_wrapper_routed.dcp"
}

puts "=== opening checkpoint: $dcp ==="
open_checkpoint $dcp

proc walk_back_to_seq {start_pin max_depth} {
    # Walks backward from a load pin looking for the first sequential
    # driver. Returns a list of {depth cell ref_name} for each hop, and
    # terminates when it hits an FD* / RAM* / URAM* primitive or exceeds
    # max_depth. Treats LUTs / MUXF* / CARRY as combinational pass-through.
    set visited [dict create]
    set frontier [list [list 0 $start_pin]]
    set results [list]
    while {[llength $frontier] > 0} {
        set entry [lindex $frontier 0]
        set frontier [lrange $frontier 1 end]
        set depth [lindex $entry 0]
        set pin   [lindex $entry 1]
        if {$depth > $max_depth} { continue }
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set src_pins [get_pins -of $net -filter {DIRECTION == OUT} -quiet]
        foreach sp $src_pins {
            set scell [get_cells -of $sp -quiet]
            if {[llength $scell] == 0} { continue }
            set sref [get_property REF_NAME $scell]
            set key "$scell"
            if {[dict exists $visited $key]} { continue }
            dict set visited $key 1
            if {[string match "FD*" $sref] || [string match "URAM*" $sref] || [string match "RAMB*" $sref]} {
                lappend results [list $depth $scell $sref $sp]
                continue
            }
            # Comb pass-through - recurse backward from all its inputs
            set in_pins [get_pins -of $scell -filter {DIRECTION == IN} -quiet]
            foreach ip $in_pins {
                lappend frontier [list [expr {$depth + 1}] $ip]
            }
        }
    }
    return $results
}

proc walk_fwd_to_seq {start_pin max_depth} {
    # Walk forward from an OUT pin until hitting the first sequential
    # load (FD*/URAM*/RAMB*) on each branch.
    set visited [dict create]
    set frontier [list [list 0 $start_pin]]
    set results [list]
    while {[llength $frontier] > 0} {
        set entry [lindex $frontier 0]
        set frontier [lrange $frontier 1 end]
        set depth [lindex $entry 0]
        set pin   [lindex $entry 1]
        if {$depth > $max_depth} { continue }
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set load_pins [get_pins -of $net -filter {DIRECTION == IN} -quiet]
        foreach lp $load_pins {
            set lcell [get_cells -of $lp -quiet]
            if {[llength $lcell] == 0} { continue }
            set lref [get_property REF_NAME $lcell]
            set key "$lcell|[get_property NAME $lp]"
            if {[dict exists $visited $key]} { continue }
            dict set visited $key 1
            if {[string match "FD*" $lref] || [string match "URAM*" $lref] || [string match "RAMB*" $lref]} {
                lappend results [list $depth $lcell $lref $lp]
                continue
            }
            # Comb pass-through - recurse forward from all its outputs
            set out_pins [get_pins -of $lcell -filter {DIRECTION == OUT} -quiet]
            foreach op $out_pins {
                lappend frontier [list [expr {$depth + 1}] $op]
            }
        }
    }
    return $results
}

puts "\n=== finding URAM288 cells under u_fmap_a / u_fmap_b ==="

set pattern_list {
    {*u_fmap_a*}
    {*u_fmap_b*}
    {*fmap_a*}
    {*fmap_b*}
}

set fmap_urams [list]
foreach pat $pattern_list {
    set hits [get_cells -hierarchical -filter "REF_NAME == URAM288 && NAME =~ $pat" -quiet]
    foreach h $hits {
        if {[lsearch -exact $fmap_urams $h] < 0} {
            lappend fmap_urams $h
        }
    }
}

if {[llength $fmap_urams] == 0} {
    puts "WARNING: no fmap-prefixed URAM cells found. Listing ALL URAM288 cells:"
    set fmap_urams [get_cells -hierarchical -filter {REF_NAME == URAM288} -quiet]
}

puts "  found [llength $fmap_urams] URAM cells to inspect"

# Split into fmap_a and fmap_b lists for clarity
set urams_a [list]
set urams_b [list]
foreach c $fmap_urams {
    if {[string first "u_fmap_a" $c] >= 0 || [string first "fmap_a"   $c] >= 0} {
        lappend urams_a $c
    } elseif {[string first "u_fmap_b" $c] >= 0 || [string first "fmap_b" $c] >= 0} {
        lappend urams_b $c
    }
}
puts "  u_fmap_a URAMs: [llength $urams_a]"
puts "  u_fmap_b URAMs: [llength $urams_b]"

# --------------------------------------------------------------------
# Detailed address + data path trace - pick ONE URAM from each ram to
# reduce noise (they are structurally identical under bit-slice cascade)
# --------------------------------------------------------------------
proc inspect_uram_ports {uram label} {
    puts ""
    puts "=============================================================="
    puts "== $label : $uram"
    puts "=============================================================="

    set addr_pins [get_pins -of $uram -filter {DIRECTION == IN && NAME =~ *ADDR_B*} -quiet]
    puts "  ADDR_B pin count: [llength $addr_pins]"

    set addr_flops [dict create]
    set addr_direct_flop_count 0
    foreach pin $addr_pins {
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set src_pins [get_pins -of $net -filter {DIRECTION == OUT} -quiet]
        foreach sp $src_pins {
            set scell [get_cells -of $sp -quiet]
            if {[llength $scell] == 0} { continue }
            set sref [get_property REF_NAME $scell]
            dict incr addr_flops "$sref"
            if {[string match "FD*" $sref]} {
                incr addr_direct_flop_count
            }
        }
    }
    puts "  ADDR_B immediate driver ref types:"
    foreach {k v} [dict get $addr_flops] {
        puts "    $k : $v"
    }

    # Walk BACK from one ADDR_B pin to the nearest sequential driver(s)
    if {[llength $addr_pins] > 0} {
        set sample [lindex $addr_pins 0]
        puts ""
        puts "  backward walk from [get_property NAME $sample] (max 8 hops)"
        set back [walk_back_to_seq $sample 8]
        foreach r $back {
            set d [lindex $r 0]
            set cell [lindex $r 1]
            set ref  [lindex $r 2]
            puts "    depth=$d  $ref  $cell"
        }
    }

    set dout_pins [get_pins -of $uram -filter {DIRECTION == OUT && NAME =~ *DOUT_B*} -quiet]
    puts ""
    puts "  DOUT_B pin count: [llength $dout_pins]"
    set dout_loads [dict create]
    set dout_direct_flop_count 0
    foreach pin $dout_pins {
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set load_pins [get_pins -of $net -filter {DIRECTION == IN} -quiet]
        foreach lp $load_pins {
            set lcell [get_cells -of $lp -quiet]
            if {[llength $lcell] == 0} { continue }
            set lref [get_property REF_NAME $lcell]
            dict incr dout_loads "$lref"
            if {[string match "FD*" $lref]} {
                incr dout_direct_flop_count
            }
        }
    }
    puts "  DOUT_B immediate load ref types:"
    foreach {k v} [dict get $dout_loads] {
        puts "    $k : $v"
    }

    # Walk FORWARD from one DOUT_B pin to the nearest sequential load(s)
    if {[llength $dout_pins] > 0} {
        set sample [lindex $dout_pins 0]
        puts ""
        puts "  forward walk from [get_property NAME $sample] (max 8 hops)"
        set fwd [walk_fwd_to_seq $sample 8]
        # Take at most 12 results
        set count 0
        foreach r $fwd {
            if {$count >= 12} { break }
            set d [lindex $r 0]
            set cell [lindex $r 1]
            set ref  [lindex $r 2]
            puts "    depth=$d  $ref  $cell"
            incr count
        }
        puts "    ([llength $fwd] total sequential loads reached)"
    }

    puts ""
    puts "  SUMMARY for $label:"
    puts "    direct FD* drivers on ADDR_B pins: $addr_direct_flop_count / [llength $addr_pins]"
    puts "    direct FD* loads    on DOUT_B pins: $dout_direct_flop_count / [llength $dout_pins]"
}

# Inspect one representative URAM from each bank
if {[llength $urams_a] > 0} {
    inspect_uram_ports [lindex $urams_a 0] "fmap_a representative #0"
}
if {[llength $urams_a] > 1} {
    inspect_uram_ports [lindex $urams_a end] "fmap_a representative #end"
}
if {[llength $urams_b] > 0} {
    inspect_uram_ports [lindex $urams_b 0] "fmap_b representative #0"
}
if {[llength $urams_b] > 1} {
    inspect_uram_ports [lindex $urams_b end] "fmap_b representative #end"
}

# --------------------------------------------------------------------
# Aggregate: across ALL fmap URAM cells, count how many have FD* direct
# drivers on ADDR_B and FD* direct loads on DOUT_B. Any outlier
# indicates a hidden register.
# --------------------------------------------------------------------
puts ""
puts "=============================================================="
puts "== AGGREGATE SCAN OVER ALL fmap URAM CELLS"
puts "=============================================================="
set tot_addr_pins 0
set tot_addr_fd   0
set tot_addr_uram 0
set tot_dout_pins 0
set tot_dout_fd   0
set tot_dout_uram 0
set tot_dout_mux  0
foreach c $fmap_urams {
    set addr_pins [get_pins -of $c -filter {DIRECTION == IN && NAME =~ *ADDR_B*} -quiet]
    foreach pin $addr_pins {
        incr tot_addr_pins
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set src_pins [get_pins -of $net -filter {DIRECTION == OUT} -quiet]
        foreach sp $src_pins {
            set scell [get_cells -of $sp -quiet]
            if {[llength $scell] == 0} { continue }
            set sref [get_property REF_NAME $scell]
            if {[string match "FD*" $sref]}   { incr tot_addr_fd }
            if {[string match "URAM*" $sref]} { incr tot_addr_uram }
        }
    }
    set dout_pins [get_pins -of $c -filter {DIRECTION == OUT && NAME =~ *DOUT_B*} -quiet]
    foreach pin $dout_pins {
        incr tot_dout_pins
        set net [get_nets -of $pin -quiet]
        if {[llength $net] == 0} { continue }
        set lp [get_pins -of $net -filter {DIRECTION == IN} -quiet]
        foreach l $lp {
            set lcell [get_cells -of $l -quiet]
            if {[llength $lcell] == 0} { continue }
            set lref [get_property REF_NAME $lcell]
            if {[string match "FD*" $lref]}   { incr tot_dout_fd }
            if {[string match "URAM*" $lref]} { incr tot_dout_uram }
            if {[string match "MUX*" $lref] || [string match "LUT*" $lref]} { incr tot_dout_mux }
        }
    }
}
puts "  ADDR_B total input pins inspected : $tot_addr_pins"
puts "    immediate FD* drivers            : $tot_addr_fd"
puts "    immediate URAM*  drivers         : $tot_addr_uram"
puts "  DOUT_B total output pins inspected : $tot_dout_pins"
puts "    immediate FD*  loads             : $tot_dout_fd"
puts "    immediate URAM* loads (cascade)  : $tot_dout_uram"
puts "    immediate LUT/MUX loads          : $tot_dout_mux"

# --------------------------------------------------------------------
# Timing-based cross-check: worst setup path into any fmap URAM ADDR_B
# and worst launch path out of any fmap URAM DOUT_B. If there are ZERO
# paths from a flop into ADDR_B whose slack is > clock period (i.e. it
# would be multi-cycle), the read is 1-cycle. Conversely, if the path
# launches from a *different* flop than conv3d's pixel_bram_addr, a
# hidden flop was inserted.
# --------------------------------------------------------------------
puts ""
puts "=============================================================="
puts "== TIMING REPORT: into ADDR_B on fmap URAM cells"
puts "=============================================================="
set addr_pin_set [get_pins -of $fmap_urams -filter {DIRECTION == IN && NAME =~ *ADDR_B*} -quiet]
if {[llength $addr_pin_set] > 0} {
    report_timing -to $addr_pin_set -max_paths 4 -nworst 4 -path_type full -input_pins
}

puts ""
puts "=============================================================="
puts "== TIMING REPORT: out of DOUT_B on fmap URAM cells"
puts "=============================================================="
set dout_pin_set [get_pins -of $fmap_urams -filter {DIRECTION == OUT && NAME =~ *DOUT_B*} -quiet]
if {[llength $dout_pin_set] > 0} {
    report_timing -from $dout_pin_set -max_paths 4 -nworst 4 -path_type full -input_pins
}

puts ""
puts "=== done ==="
close_design
exit
