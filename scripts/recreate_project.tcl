# scripts/recreate_project.tcl
#
# Recreates tinyissimoyolo.xpr from tracked source files + block designs.
# Eliminates .xpr merge conflicts by treating the XPR as a generated artifact.
#
# The block design (.bd) files are the single source of truth for the
# system architecture.  This script builds the project container around
# them: adds RTL sources, weight ROMs, testbenches, and constraints, then
# runs generate_target so Vivado resolves all BD IP instances and creates
# the .xci + wrapper files on the fly.
#
# Usage:
#   vivado -mode batch -source scripts/recreate_project.tcl
#
# Can also be sourced from other Tcl scripts (e.g. rebuild_bitstream.tcl).
# Ends with close_project, NOT exit.
#
# Prerequisites:
#   - hardware/ip_repo/ must contain a packaged IP with component.xml
#     (run sync-ip-src.sh + package_accelerator_ip.tcl first)
#   - playground.bd must exist at its canonical location under .srcs/
#
# Run from the repo root.

puts "\n=== recreate_project.tcl ==="

set repo_root [file normalize [pwd]]
set proj_dir  $repo_root/hardware/vivado/tinyissimoyolo
set proj_name tinyissimoyolo

# ─── Pre-check: IP repo must be populated ─────────────────────────────
set component_files [concat \
    [glob -nocomplain $repo_root/hardware/ip_repo/*/component.xml] \
    [glob -nocomplain $repo_root/hardware/ip_repo/component.xml] \
]
if {[llength $component_files] == 0} {
    puts "ERROR: No component.xml found in hardware/ip_repo/"
    puts "       Run these first:"
    puts "         bash scripts/sync-ip-src.sh"
    puts "         vivado -mode batch -source hardware/scripts/package_accelerator_ip.tcl"
    error "IP repo not populated"
}
puts "  Found IP: [lindex $component_files 0]"

# ─── Pre-check: BD file must exist ────────────────────────────────────
set bd_src $proj_dir/$proj_name.srcs/sources_1/bd/playground/playground.bd
if {![file exists $bd_src]} {
    puts "ERROR: playground.bd not found at $bd_src"
    puts "       This file should be tracked in git."
    error "Block design not found"
}

# ─── Guard: refuse to clobber an open project ─────────────────────────
set lock_files [glob -nocomplain $proj_dir/.Xil/Vivado-*-lock]
if {[llength $lock_files] > 0} {
    puts "ERROR: Vivado appears to have this project open (lock file found)."
    puts "       Close Vivado first, then re-run."
    error "Project is locked by another Vivado instance"
}

if {[file exists $proj_dir/$proj_name.xpr]} {
    puts "  Removing existing $proj_name.xpr ..."
    file delete -force $proj_dir/$proj_name.xpr
}

# ─── Clean transient directories ──────────────────────────────────────
foreach d {.cache .gen .runs .hw .sim .ip_user_files} {
    set path $proj_dir/$proj_name$d
    if {[file exists $path]} {
        puts "  Cleaning $d ..."
        file delete -force $path
    }
}

# ─── 1. Create project ────────────────────────────────────────────────
puts "\n=== Creating project ==="
create_project $proj_name $proj_dir -part xck26-sfvc784-2LV-c -force
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
set_property target_language Verilog [current_project]

# ─── 2. IP repo path ──────────────────────────────────────────────────
set_property ip_repo_paths [list $repo_root/hardware/ip_repo] [current_project]
update_ip_catalog -rebuild

# ─── 3. Add RTL sources ───────────────────────────────────────────────
# Glob all .v/.sv in hardware/rtl/ — this is future-proof: when HLS
# re-exports new files, they appear automatically.
set rtl_v  [glob -nocomplain $repo_root/hardware/rtl/*.v]
set rtl_sv [glob -nocomplain $repo_root/hardware/rtl/*.sv]
set rtl_all [concat $rtl_v $rtl_sv]
# Filter out .svh (added separately as headers) and README
set rtl_files [list]
foreach f $rtl_all {
    set base [file tail $f]
    if {[string match "*.svh" $base]} continue
    if {[string match "README*" $base]} continue
    lappend rtl_files $f
}
if {[llength $rtl_files] > 0} {
    add_files -norecurse -fileset sources_1 $rtl_files
}
puts "  Added [llength $rtl_files] RTL source files"

# ─── 4. Add global include header ─────────────────────────────────────
set svh_file $repo_root/hardware/weights/hdl/layer_config.svh
if {[file exists $svh_file]} {
    add_files -norecurse -fileset sources_1 $svh_file
    set_property file_type "Verilog Header" \
        [get_files -of_objects [get_filesets sources_1] */layer_config.svh]
    set_property is_global_include true \
        [get_files -of_objects [get_filesets sources_1] */layer_config.svh]
    puts "  Added layer_config.svh as global include"
}

# ─── 5. Add weight/ROM .mem and HLS .dat files ────────────────────────
set mem_files [list]
foreach name {weight_rom.mem qp_packed_rom.mem silu_lut.mem zp_in_rom.mem zp_out_rom.mem} {
    set f $repo_root/hardware/weights/hdl/$name
    if {[file exists $f]} { lappend mem_files $f }
}
if {[llength $mem_files] > 0} {
    add_files -norecurse -fileset sources_1 $mem_files
    puts "  Added [llength $mem_files] weight .mem files"
}

set dat_files [glob -nocomplain $repo_root/hardware/rtl/*.dat]
if {[llength $dat_files] > 0} {
    add_files -norecurse -fileset sources_1 $dat_files
    puts "  Added [llength $dat_files] HLS .dat ROM init files"
}

# ─── 6. Add block designs ─────────────────────────────────────────────
# playground.bd must live inside the project's .srcs/ tree (the BD's
# internal gen_directory field uses a relative path).  It is tracked
# in git at this exact location.
add_files -norecurse -fileset sources_1 $bd_src
puts "  Added playground.bd"

# tinyissimoyolo.bd — disabled secondary camera pipeline BD
set bd2_src $proj_dir/$proj_name.srcs/sources_1/bd/tinyissimoyolo/tinyissimoyolo.bd
if {[file exists $bd2_src]} {
    add_files -norecurse -fileset sources_1 $bd2_src
    set_property USER_DISABLED 1 [get_files */tinyissimoyolo.bd]
    puts "  Added tinyissimoyolo.bd (disabled)"
}

# ─── 7. Generate BD output products ───────────────────────────────────
# This is the critical step: Vivado reads the BD JSON, resolves all IP
# references against the catalog, and generates the .xci files + wrappers.
# The Zynq PS config is embedded in the BD JSON, so it comes out
# correctly with no extra Tcl.
puts "\n=== Generating output products for playground.bd ==="
set bd_file [get_files playground.bd -filter {IS_BLOCK_DESIGN == 1}]
generate_target all $bd_file
puts "  BD output products generated"

# Create the HDL wrapper for the BD
make_wrapper -files $bd_file -top
set wrapper_file [glob -nocomplain \
    $proj_dir/$proj_name.gen/sources_1/bd/playground/hdl/playground_wrapper.v]
if {$wrapper_file ne ""} {
    add_files -norecurse -fileset sources_1 $wrapper_file
    puts "  Added playground_wrapper.v"
}

# ─── 8. Add constraints ───────────────────────────────────────────────
set xdc_files [glob -nocomplain $repo_root/hardware/constraints/*.xdc]
foreach xdc $xdc_files {
    add_files -norecurse -fileset constrs_1 $xdc
    # Camera XDC is disabled by default (only used with camera pipeline)
    if {[string match "*camera*" [file tail $xdc]]} {
        set_property USER_DISABLED 1 [get_files */[file tail $xdc]]
    }
    puts "  Added constraint: [file tail $xdc]"
}

# ─── 9. Add simulation sources ────────────────────────────────────────
set tb_files [glob -nocomplain $repo_root/hardware/testbench/*.sv]
if {[llength $tb_files] > 0} {
    add_files -norecurse -fileset sim_1 $tb_files
    puts "  Added [llength $tb_files] testbench files"
}

# Simulation data files (.mem) — golden references + test inputs
set sim_mem_files [glob -nocomplain $repo_root/hardware/testbench/inference_hdl/*.mem]
if {[llength $sim_mem_files] > 0} {
    add_files -norecurse -fileset sim_1 $sim_mem_files
    puts "  Added [llength $sim_mem_files] simulation .mem data files"
}

# Add layer_config.svh to sim fileset too (needed by testbenches)
if {[file exists $svh_file]} {
    add_files -norecurse -fileset sim_1 $svh_file
    set_property file_type "Verilog Header" \
        [get_files -of_objects [get_filesets sim_1] */layer_config.svh]
    set_property is_global_include true \
        [get_files -of_objects [get_filesets sim_1] */layer_config.svh]
}

# ─── 10. Set design tops ──────────────────────────────────────────────
set_property top playground_wrapper [current_fileset]
set_property top tb_tinyissimoyolo_accel [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# ─── 11. Configure implementation strategy ────────────────────────────
# Hold-fix insurance: -tns_cleanup runs an extra hold-fix pass at the
# end of route_design that inserts LUT1 buffers on paths with negative
# or near-zero hold slack.  Essential for the conv3d ACC path.
set_property -name {STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS} \
    -value {-tns_cleanup} -objects [get_runs impl_1]
puts "  Set impl_1 route_design -tns_cleanup"

# ─── 12. Validate the BD ──────────────────────────────────────────────
puts "\n=== Validating block design ==="
open_bd_design $bd_file
validate_bd_design -force
save_bd_design
close_bd_design [current_bd_design]
puts "  Block design validated OK"

# ─── Done ──────────────────────────────────────────────────────────────
puts "\n=========================================="
puts " Project recreated successfully:"
puts "   $proj_dir/$proj_name.xpr"
puts "=========================================="
puts ""
puts "Next steps:"
puts "  GUI:       vivado $proj_dir/$proj_name.xpr"
puts "  Build:     vivado -mode batch -source scripts/rebuild_bitstream.tcl"
puts "  Simulate:  vivado -mode batch -source scripts/run_sim_headless.tcl"
puts ""

close_project
