# package_accelerator_ip.tcl
#
# Builds a self-contained Vivado packaging project that pulls together:
#   - all RTL needed by tinyissimoyolo_accelerator (HDL + HLS dual engine)
#   - layer_config.svh as a global include
#   - the three pre-baked HDL ROM .mem init files
#   - all 41 Vitis-HLS-generated .v files + 14 .dat ROM init files
#
# ...then runs ipx::package_project to emit a packaged IP at
#   hardware/ip_repo/tinyissimoyolo_accelerator_v1_0/
#
# After running this script, point your main project at the new IP repo:
#   set_property ip_repo_paths {./hardware/ip_repo} [current_project]
#   update_ip_catalog
# ...and instantiate `tinyissimoyolo_accelerator_v1.0` in your BD.
#
# Run from the repo root:
#   vivado -mode batch -source hardware/scripts/package_accelerator_ip.tcl
#
# Idempotent: -force re-creates the packaging project from scratch each
# time, so re-running after RTL or HLS regen is safe.

set repo_root  [file normalize [pwd]]
set pkg_dir    $repo_root/hardware/vivado/accelerator_pkg
set ip_out_dir $repo_root/hardware/ip_repo/tinyissimoyolo_accelerator_v1_0
set rtl_dir    $repo_root/hardware/rtl
set hdr_dir    $repo_root/hardware/weights/hdl
set hls_dir    $repo_root/hardware/hls/tinyissimo_layer/tinyissimo_layer/hls/impl/ip/hdl/verilog

if {![file isdirectory $hls_dir]} {
    puts "ERROR: $hls_dir not found — re-run Vitis HLS export to populate it."
    exit 1
}

# Create the packaging project (-force replaces any prior one)
puts "\n=== Creating packaging project at $pkg_dir ==="
create_project tinyissimoyolo_accel_pkg $pkg_dir \
    -part xck26-sfvc784-2LV-c -force

set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
set_property target_language Verilog [current_project]

# ─────────────────────────────────────────────────────────────────────
# Core RTL (HDL + HLS wrapper + IP outermost wrapper)
# ─────────────────────────────────────────────────────────────────────
set rtl_files [list \
    $rtl_dir/sdp_ram.sv \
    $rtl_dir/circular_buffer.v \
    $rtl_dir/mac_manual.v \
    $rtl_dir/convolver.v \
    $rtl_dir/conv3d.v \
    $rtl_dir/conv1d.v \
    $rtl_dir/activation.sv \
    $rtl_dir/max_pool.sv \
    $rtl_dir/axil_regs.sv \
    $rtl_dir/inference_hdl.sv \
    $rtl_dir/inference_hls.sv \
    $rtl_dir/inference_top.sv \
    $rtl_dir/tinyissimoyolo_accelerator.sv \
]
puts "\n=== Adding [llength $rtl_files] core RTL files ==="
add_files -norecurse -fileset sources_1 $rtl_files

# layer_config.svh — needed as a global include because inference_top
# and inference_hdl both use \`include "layer_config.svh"
puts "\n=== Adding layer_config.svh as global include ==="
add_files -norecurse -fileset sources_1 $hdr_dir/layer_config.svh
set_property file_type "Verilog Header" [get_files $hdr_dir/layer_config.svh]
set_property is_global_include true     [get_files $hdr_dir/layer_config.svh]

# HDL ROM .mem files (referenced by sdp_ram MEM_FILE parameter)
puts "\n=== Adding HDL ROM .mem init files ==="
add_files -norecurse -fileset sources_1 [list \
    $hdr_dir/weight_rom.mem \
    $hdr_dir/qp_packed_rom.mem \
    $hdr_dir/silu_lut.mem \
    $hdr_dir/zp_in_rom.mem \
    $hdr_dir/zp_out_rom.mem \
]

# ─────────────────────────────────────────────────────────────────────
# HLS-generated sources
# ─────────────────────────────────────────────────────────────────────
set hls_v   [lsort [glob -nocomplain $hls_dir/*.v]]
set hls_dat [lsort [glob -nocomplain $hls_dir/*.dat]]
puts "\n=== Adding [llength $hls_v] HLS .v + [llength $hls_dat] HLS .dat files ==="
add_files -norecurse -fileset sources_1 [concat $hls_v $hls_dat]

# Top
update_compile_order -fileset sources_1
set_property top tinyissimoyolo_accelerator [current_fileset]
update_compile_order -fileset sources_1
puts "\n=== Top set to [get_property top [current_fileset]] ==="

# Quick elaboration check before packaging — catches missing files /
# undefined modules / port mismatches early.
puts "\n=== Elaborating to validate the design ==="
synth_design -rtl -name rtl_check -mode out_of_context
puts "Elaboration OK."

# ─────────────────────────────────────────────────────────────────────
# Package as IP
# ─────────────────────────────────────────────────────────────────────
file mkdir $ip_out_dir
puts "\n=== Packaging IP into $ip_out_dir ==="
ipx::package_project -root_dir $ip_out_dir \
    -vendor user.org -library user -taxonomy /UserIP \
    -import_files -force

set core [ipx::current_core]
set_property name           tinyissimoyolo_accelerator $core
set_property version        1.0                        $core
set_property display_name   "TinyissimoYOLO Accelerator (HDL+HLS dual engine)" $core
set_property description    "17-layer TinyissimoYOLO inference accelerator with both HDL and Vitis-HLS pipelines on silicon. AXI-Lite control + AXI-Stream pixel ingress + level-high interrupt; runtime engine select via MODE[4]." $core
set_property vendor_display_name "EE4218 Project" $core
set_property company_url    "https://github.com/" $core

# Auto-infer AXI4-Lite and AXI-Stream interfaces from the standard
# port-name conventions on tinyissimoyolo_accelerator.
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0  $core
ipx::associate_bus_interfaces -busif s_axi_lite -clock aclk  $core
ipx::associate_bus_interfaces -busif s_axis     -clock aclk  $core

# Tag the inferred aclk clock interface with FREQ_HZ so [IP_Flow 19-11770]
# is silenced and consumers see a sensible default of 100 MHz.
#
# CRITICAL: mark the parameter as user-resolvable (value_resolve_type
# = user). Without this, the IP locks FREQ_HZ to exactly 100,000,000,
# and BD validation fails when the source clock is anything else:
#
#   [BD 41-237] Bus Interface property FREQ_HZ does not match between
#               /tinyissimoyolo_accel_0/s_axi_lite(100000000) and
#               /axi_smc/M00_AXI(99999001)
#   [BD 41-238] Port/Pin property FREQ_HZ does not match between
#               /tinyissimoyolo_accel_0/aclk(100000000) and
#               /zynq_ultra_ps_e_0/pl_clk0(99999001)
#
# (The Kria K26 PS PL clock runs at 99,999,001 Hz, not exactly 100 MHz —
# the PLL divider chain off the 33.333 MHz reference doesn't land
# precisely on 100M.)
#
# With value_resolve_type=user, BD connection automation overrides the
# 100 MHz default with whatever the source clock actually provides.
ipx::add_bus_parameter FREQ_HZ \
    [ipx::get_bus_interfaces aclk -of_objects $core]
set_property value_resolve_type user \
    [ipx::get_bus_parameters FREQ_HZ \
        -of_objects [ipx::get_bus_interfaces aclk -of_objects $core]]
set_property value 100000000 \
    [ipx::get_bus_parameters FREQ_HZ \
        -of_objects [ipx::get_bus_interfaces aclk -of_objects $core]]

# Mark irq_done as a Xilinx interrupt sideband.
#
# The Xilinx interrupt interface lives under the "signal:" namespace,
# NOT "interface:" — using the wrong VLNV produces:
#   [IP_Flow 19-569] Cannot find bus abstraction file ...
# SENSITIVITY = LEVEL_HIGH because the FSM holds irq_done high during
# the entire PH_DONE state (see inference_top.sv:306, irq_done assigned
# from a level-sensitive phase comparison, not edge-detected).
ipx::add_bus_interface interrupt $core
set_property interface_mode master [ipx::get_bus_interfaces interrupt -of_objects $core]
set_property abstraction_type_vlnv xilinx.com:signal:interrupt_rtl:1.0 \
    [ipx::get_bus_interfaces interrupt -of_objects $core]
set_property bus_type_vlnv xilinx.com:signal:interrupt:1.0 \
    [ipx::get_bus_interfaces interrupt -of_objects $core]
ipx::add_port_map INTERRUPT [ipx::get_bus_interfaces interrupt -of_objects $core]
set_property physical_name irq_done \
    [ipx::get_port_maps INTERRUPT -of_objects \
        [ipx::get_bus_interfaces interrupt -of_objects $core]]
ipx::add_bus_parameter SENSITIVITY \
    [ipx::get_bus_interfaces interrupt -of_objects $core]
set_property value LEVEL_HIGH \
    [ipx::get_bus_parameters SENSITIVITY \
        -of_objects [ipx::get_bus_interfaces interrupt -of_objects $core]]

ipx::create_xgui_files $core
ipx::update_checksums  $core
ipx::check_integrity   $core
ipx::save_core         $core

puts "\n=== Done ==="
puts "Packaged IP: $ip_out_dir/component.xml"
puts ""
puts "Next steps in your main project (tinyissimoyolo.xpr):"
puts "  set_property ip_repo_paths {./hardware/ip_repo} \[current_project\]"
puts "  update_ip_catalog"
puts "  # Then add tinyissimoyolo_accelerator to your BD via the IP catalog."

close_project
exit
