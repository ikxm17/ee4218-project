# add_hls_sources_to_xpr.tcl
#
# Adds the Vitis-HLS-generated Verilog sources (and the .dat ROM init
# files they reference via $readmemh) to the existing tinyissimoyolo.xpr
# project as design sources, by reference (no import / no copy).
#
# Why "by reference": each *_ROM_AUTO_1R.v contains
#   $readmemh("./tinyissimo_layer_top_LAYER_CFG_*_ROM_AUTO_1R.dat", rom0)
# which is a path RELATIVE to the .v file's directory.  As long as we do
# NOT import the .v files into the project's local sources/imports tree
# (default behaviour of `add_files` is import-on-add only when the project
# was created with -import_files; here we use plain `add_files`), the .v
# files stay in hardware/hls/.../impl/ip/hdl/verilog/ next to their .dat
# siblings, and $readmemh resolves cleanly during both simulation and
# synthesis.
#
# Run from the repo root:
#   vivado -mode batch -source hardware/scripts/add_hls_sources_to_xpr.tcl
# Or paste into the Vivado Tcl console with the project already open.

set repo_root [file normalize [pwd]]
set xpr_path  $repo_root/hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr
set hls_dir   $repo_root/hardware/hls/tinyissimo_layer/tinyissimo_layer/hls/impl/ip/hdl/verilog

if {![file exists $xpr_path]} {
    puts "ERROR: $xpr_path not found — run from repo root."
    exit 1
}
if {![file isdirectory $hls_dir]} {
    puts "ERROR: $hls_dir not found — re-run Vitis HLS export to populate it."
    exit 1
}

# Open existing project (skip if already open via the Tcl console)
if {[catch {current_project} _]} {
    open_project $xpr_path
}

# Glob the HLS-generated files
set hls_v_files   [lsort [glob -nocomplain $hls_dir/*.v]]
set hls_dat_files [lsort [glob -nocomplain $hls_dir/*.dat]]

puts "Found [llength $hls_v_files] .v and [llength $hls_dat_files] .dat HLS files."

# Add (skip files already in the project)
set added 0
set skipped 0
foreach f [concat $hls_v_files $hls_dat_files] {
    if {[llength [get_files -quiet $f]] > 0} {
        incr skipped
        continue
    }
    add_files -norecurse -fileset sources_1 $f
    incr added
}

puts "Added $added file(s); $skipped already in the project."

# Refresh compile order so the new sources are picked up
update_compile_order -fileset sources_1

# Save the xpr
save_project_as [current_project] -force

puts "Done.  HLS sources are now visible under sources_1."
puts "Note: this script does NOT change the top module or remove any"
puts "      stale RTL references in the existing project — if you want"
puts "      to sim/synth the new dual-engine inference_top.sv here, you"
puts "      will also need to add inference_top.sv, inference_hls.sv,"
puts "      axil_regs.sv, conv1d.v, circular_buffer.v, and"
puts "      tinyissimoyolo_accelerator.sv, and remove top.sv,"
puts "      variable_shift_register.v, memory_ram.v.  See"
puts "      hardware/scripts/package_accelerator_ip.tcl for the canonical"
puts "      file list."
