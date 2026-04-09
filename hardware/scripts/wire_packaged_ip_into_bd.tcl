# wire_packaged_ip_into_bd.tcl
#
# Wires the freshly-packaged accelerator IP at
#   hardware/ip_repo/tinyissimoyolo_accelerator_v1_0/
# into the existing playground.bd inside tinyissimoyolo.xpr.
#
# Why this works without touching the BD itself:
#
#   The BD's existing cell `tinyissimoyolo_accel_0` already references
#   VLNV `user.org:user:tinyissimoyolo_accelerator:1.0` — exactly the
#   same VLNV the new packaging emits. The interface names on the IP
#   (s_axi_lite, s_axis, aclk, aresetn, irq_done) are unchanged, so
#   the BD connections remain valid. All we need to do is:
#     1. Point the project's IP catalog at hardware/ip_repo/
#     2. update_ip_catalog so Vivado discovers the new sources
#     3. reset + re-generate the BD output products so the per-instance
#        sources under tinyissimoyolo.gen/.../bd/playground/ip/ get
#        rebuilt from the NEW catalog entry instead of the cached
#        ones from whenever the IP was last packaged
#     4. validate_bd_design to confirm everything still wires up
#
# Run from the repo root:
#   vivado -mode batch -source hardware/scripts/wire_packaged_ip_into_bd.tcl
#
# Idempotent: safe to re-run after any IP repackaging.

set repo_root [file normalize [pwd]]
set xpr_path  $repo_root/hardware/vivado/tinyissimoyolo/tinyissimoyolo.xpr
set ip_repo   $repo_root/hardware/ip_repo

if {![file exists $xpr_path]} {
    puts "ERROR: $xpr_path not found — run from repo root."
    exit 1
}
if {![file exists $ip_repo/tinyissimoyolo_accelerator_v1_0/component.xml]} {
    puts "ERROR: packaged IP not found at"
    puts "       $ip_repo/tinyissimoyolo_accelerator_v1_0/component.xml"
    puts "       Run hardware/scripts/package_accelerator_ip.tcl first."
    exit 1
}

open_project $xpr_path

# ─────────────────────────────────────────────────────────────────────
# 1. Add hardware/ip_repo to ip_repo_paths (idempotent)
# ─────────────────────────────────────────────────────────────────────
set existing_paths [get_property ip_repo_paths [current_project]]
puts "\n=== Current ip_repo_paths: $existing_paths ==="
if {[lsearch -exact $existing_paths $ip_repo] < 0} {
    set new_paths [concat $existing_paths $ip_repo]
    set_property ip_repo_paths $new_paths [current_project]
    puts "Added $ip_repo"
} else {
    puts "Already present"
}

# ─────────────────────────────────────────────────────────────────────
# 2. Refresh the IP catalog so Vivado discovers the new component.xml
# ─────────────────────────────────────────────────────────────────────
puts "\n=== Updating IP catalog ==="
update_ip_catalog
update_ip_catalog -rebuild

# ─────────────────────────────────────────────────────────────────────
# 3. Status report (informational)
# ─────────────────────────────────────────────────────────────────────
puts "\n=== IP status before regeneration ==="
report_ip_status -name ip_status_before

set accel_ips [get_ips -filter {IPDEF =~ "user.org:user:tinyissimoyolo_accelerator:*"}]
puts "Accelerator IP instances in project: $accel_ips"

# ─────────────────────────────────────────────────────────────────────
# 4. Reset + re-generate BD output products so the per-instance
#    sources are pulled from the NEW IP catalog entry
# ─────────────────────────────────────────────────────────────────────
set bd_file [get_files playground.bd]
puts "\n=== BD file: $bd_file ==="

puts "\n=== Resetting BD output products ==="
reset_target {synthesis simulation} $bd_file -quiet

puts "\n=== Re-generating BD output products ==="
generate_target {synthesis simulation} $bd_file

# ─────────────────────────────────────────────────────────────────────
# 5. Open + validate the BD to confirm everything is still wired
# ─────────────────────────────────────────────────────────────────────
puts "\n=== Opening + validating BD ==="
open_bd_design $bd_file
validate_bd_design -force
save_bd_design

# ─────────────────────────────────────────────────────────────────────
# 6. Final status
# ─────────────────────────────────────────────────────────────────────
puts "\n=== IP status after regeneration ==="
report_ip_status -name ip_status_after

puts "\n=== Done ==="
puts ""
puts "Next steps:"
puts "  1. Quick build sanity check:"
puts "       vivado -mode batch -source scripts/rebuild_bitstream.tcl"
puts ""
puts "  2. Or open the project and inspect the BD:"
puts "       vivado $xpr_path"
puts "       (open playground.bd, confirm tinyissimoyolo_accel_0 cell"
puts "        shows v1.0 with the new ports/registers)"

close_project
exit
