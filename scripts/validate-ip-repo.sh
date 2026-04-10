#!/usr/bin/env bash
#
# scripts/validate-ip-repo.sh
#
# Pre-flight check for IP repo health. Runs without Vivado.
# Catches common problems that cause synthesis failures:
#   1. Missing design files in component.xml
#   2. Polluted ip_repo/src/ (golden .mem, testbenches, BD artifacts)
#   3. Stale .dat/.v files (rtl/ newer than ip_repo/src/)
#   4. Duplicate component.xml (Vivado may pick the wrong one)
#
# Exit code: 0 = clean, 1 = problems found
#
# Usage:
#   bash scripts/validate-ip-repo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RTL_DIR="$REPO_ROOT/hardware/rtl"
IP_REPO="$REPO_ROOT/hardware/ip_repo"
WEIGHTS_DIR="$REPO_ROOT/hardware/weights/hdl"

errors=0
warnings=0

pass()  { echo "  ✓ $1"; }
warn()  { echo "  ⚠ $1"; warnings=$((warnings + 1)); }
fail()  { echo "  ✗ $1"; errors=$((errors + 1)); }

echo "============================================"
echo " IP Repo Validation"
echo "============================================"
echo

# ─── 1. Locate component.xml ─────────────────────────────────────────
echo "── Check 1: component.xml existence ──"

root_xml="$IP_REPO/component.xml"
pkg_xml="$IP_REPO/tinyissimoyolo_accelerator_v1_0/component.xml"
component_xml=""

if [[ -f "$root_xml" ]] && [[ -f "$pkg_xml" ]]; then
    fail "DUPLICATE: component.xml exists at BOTH locations:"
    echo "       $root_xml"
    echo "       $pkg_xml"
    echo "       Vivado may pick the wrong one. Remove the stale copy."
    # Use the newer one for remaining checks
    if [[ "$pkg_xml" -nt "$root_xml" ]]; then
        component_xml="$pkg_xml"
        echo "       (Using newer: $pkg_xml)"
    else
        component_xml="$root_xml"
        echo "       (Using newer: $root_xml)"
    fi
elif [[ -f "$pkg_xml" ]]; then
    pass "component.xml found at $pkg_xml"
    component_xml="$pkg_xml"
elif [[ -f "$root_xml" ]]; then
    pass "component.xml found at $root_xml (legacy location)"
    component_xml="$root_xml"
else
    fail "component.xml NOT FOUND in $IP_REPO"
    echo "       Run: vivado -mode batch -source hardware/scripts/package_accelerator_ip.tcl"
    echo
    echo "============================================"
    echo " RESULT: FAIL ($((errors)) errors)"
    echo "============================================"
    exit 1
fi
echo

# ─── 2. Completeness — are all design files listed? ──────────────────
echo "── Check 2: component.xml completeness (synthesis fileset) ──"

# Extract file names from the synthesis fileset.
# The fileset is delimited by:
#   <spirit:name>xilinx_anylanguagesynthesis_view_fileset</spirit:name>
#   ... file entries ...
#   </spirit:fileSet>
#
# Within each <spirit:file> block, the filename is in
#   <spirit:name>src/foo.v</spirit:name>
# We only match entries that look like file paths (contain a dot + extension).
component_dir="$(dirname "$component_xml")"

synth_files_raw=$(sed -n '/xilinx_anylanguagesynthesis_view_fileset/,/<\/spirit:fileSet>/p' "$component_xml" \
    | grep '<spirit:name>' \
    | sed 's|.*<spirit:name>||; s|</spirit:name>.*||' \
    | grep -E '\.(v|sv|svh|mem|dat|xdc)$' \
    | sort)

# Strip path prefix (src/, hdl/, etc.) to get basenames
synth_basenames=$(echo "$synth_files_raw" | xargs -I{} basename {} | sort -u)

# Build expected file list from rtl/ (design sources only, no README)
expected_files=()
shopt -s nullglob
for f in "$RTL_DIR"/*.v "$RTL_DIR"/*.sv "$RTL_DIR"/*.dat; do
    base="$(basename "$f")"
    [[ "$base" == .* ]] && continue
    [[ "$base" == README* ]] && continue
    expected_files+=("$base")
done
# Also expect weight .mem and .svh from weights/hdl/
# These are the files the accelerator IP needs at synthesis time.
for f in "$WEIGHTS_DIR"/layer_config.svh \
         "$WEIGHTS_DIR"/weight_rom.mem \
         "$WEIGHTS_DIR"/qp_packed_rom.mem \
         "$WEIGHTS_DIR"/silu_lut.mem \
         "$WEIGHTS_DIR"/zp_in_rom.mem \
         "$WEIGHTS_DIR"/zp_out_rom.mem; do
    [[ -f "$f" ]] && expected_files+=("$(basename "$f")")
done
shopt -u nullglob

# Sort and unique
expected_sorted=$(printf '%s\n' "${expected_files[@]}" | sort -u)

# Find MISSING files (in rtl/ but not in component.xml)
missing=$(comm -23 <(echo "$expected_sorted") <(echo "$synth_basenames"))
if [[ -n "$missing" ]]; then
    count=$(echo "$missing" | wc -l)
    fail "MISSING from synthesis fileset: $count file(s)"
    echo "$missing" | while read -r f; do
        echo "       - $f"
    done
else
    pass "All $(echo "$expected_sorted" | wc -l) design files are listed in component.xml"
fi

# Find UNEXPECTED files in synthesis fileset (in component.xml but not expected)
# Filter out files that are expected extras (zp_in_rom, zp_out_rom added by packaging)
extra=$(comm -13 <(echo "$expected_sorted") <(echo "$synth_basenames"))
if [[ -n "$extra" ]]; then
    count=$(echo "$extra" | wc -l)
    warn "EXTRA in synthesis fileset: $count file(s) (may be fine if intentional)"
    echo "$extra" | while read -r f; do
        echo "       + $f"
    done
else
    pass "No unexpected files in synthesis fileset"
fi
echo

# ─── 3. Pollution — junk files in ip_repo/src/ ──────────────────────
echo "── Check 3: ip_repo/src/ pollution ──"

src_dir="$IP_REPO/src"
if [[ ! -d "$src_dir" ]]; then
    # If using the packaged subdirectory, check there
    alt_src="$component_dir/src"
    if [[ -d "$alt_src" ]]; then
        src_dir="$alt_src"
    else
        warn "No src/ directory found — skipping pollution check"
        src_dir=""
    fi
fi

if [[ -n "$src_dir" ]]; then
    pollution_found=false

    # Check for golden test files
    shopt -s nullglob
    golden_files=("$src_dir"/golden_*.mem)
    if [[ ${#golden_files[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#golden_files[@]} golden test file(s) in src/"
        for f in "${golden_files[@]}"; do
            echo "       - $(basename "$f")"
        done
        pollution_found=true
    fi

    # Check for testbench files
    tb_files=("$src_dir"/tb_*.sv "$src_dir"/tb_*.v)
    if [[ ${#tb_files[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#tb_files[@]} testbench file(s) in src/"
        for f in "${tb_files[@]}"; do
            echo "       - $(basename "$f")"
        done
        pollution_found=true
    fi

    # Check for BD artifacts
    bd_artifacts=()
    for f in "$src_dir"/bd_*.v "$src_dir"/playground*.v; do
        [[ -f "$f" ]] && bd_artifacts+=("$f")
    done
    if [[ ${#bd_artifacts[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#bd_artifacts[@]} block design artifact(s) in src/"
        for f in "${bd_artifacts[@]}"; do
            echo "       - $(basename "$f")"
        done
        pollution_found=true
    fi

    # Check for simulation-only data
    sim_data=()
    for f in "$src_dir"/pixels*.mem; do
        [[ -f "$f" ]] && sim_data+=("$f")
    done
    if [[ ${#sim_data[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#sim_data[@]} simulation data file(s) in src/"
        for f in "${sim_data[@]}"; do
            echo "       - $(basename "$f")"
        done
        pollution_found=true
    fi

    # Check for constraint files
    xdc_files=("$src_dir"/*.xdc)
    if [[ ${#xdc_files[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#xdc_files[@]} constraint file(s) in src/"
        for f in "${xdc_files[@]}"; do
            echo "       - $(basename "$f")"
        done
        pollution_found=true
    fi

    # Check for subdirectories (e.g., playground_rst_ps8_0_99M_0/)
    subdirs=()
    for d in "$src_dir"/*/; do
        [[ -d "$d" ]] && subdirs+=("$d")
    done
    if [[ ${#subdirs[@]} -gt 0 ]]; then
        fail "POLLUTION: ${#subdirs[@]} subdirectory(ies) in src/"
        for d in "${subdirs[@]}"; do
            echo "       - $(basename "$d")/"
        done
        pollution_found=true
    fi
    shopt -u nullglob

    if ! $pollution_found; then
        pass "No pollution detected in src/"
    fi
fi
echo

# ─── 4. Staleness — rtl/ newer than ip_repo/src/? ───────────────────
echo "── Check 4: file staleness ──"

if [[ -n "$src_dir" ]]; then
    stale_count=0
    missing_count=0
    shopt -s nullglob
    for rtl_file in "$RTL_DIR"/*.v "$RTL_DIR"/*.sv "$RTL_DIR"/*.dat; do
        base="$(basename "$rtl_file")"
        [[ "$base" == README* ]] && continue
        ip_file="$src_dir/$base"
        if [[ ! -f "$ip_file" ]]; then
            if (( missing_count == 0 )); then
                fail "MISSING from ip_repo/src/: files exist in rtl/ but not in ip_repo/src/"
            fi
            missing_count=$((missing_count + 1))
            echo "       - $base"
        elif ! cmp -s "$rtl_file" "$ip_file"; then
            # Content differs — this is a real staleness problem
            if (( stale_count == 0 )); then
                fail "STALE: rtl/ has different content than ip_repo/src/"
            fi
            stale_count=$((stale_count + 1))
            echo "       - $base (content differs)"
        fi
    done
    # Also check weight files
    for name in layer_config.svh weight_rom.mem qp_packed_rom.mem silu_lut.mem zp_in_rom.mem zp_out_rom.mem; do
        wt_file="$WEIGHTS_DIR/$name"
        ip_file="$src_dir/$name"
        if [[ -f "$wt_file" ]] && [[ -f "$ip_file" ]] && ! cmp -s "$wt_file" "$ip_file"; then
            if (( stale_count == 0 )); then
                fail "STALE: weights/hdl/ has different content than ip_repo/src/"
            fi
            stale_count=$((stale_count + 1))
            echo "       - $name (content differs)"
        fi
    done
    shopt -u nullglob

    if (( stale_count == 0 )); then
        pass "All ip_repo/src/ files are up to date"
    else
        echo "       Run: bash scripts/sync-ip-src.sh"
    fi
else
    warn "No src/ directory — skipping staleness check"
fi
echo

# ─── 5. component.xml vs source file check ──────────────────────────
echo "── Check 5: component.xml source file existence ──"

# Verify that every file referenced in component.xml actually exists on disk
missing_on_disk=0
while IFS= read -r relpath; do
    abspath="$component_dir/$relpath"
    if [[ ! -f "$abspath" ]]; then
        if (( missing_on_disk == 0 )); then
            fail "component.xml references files that don't exist on disk:"
        fi
        missing_on_disk=$((missing_on_disk + 1))
        echo "       - $relpath"
    fi
done < <(grep '<spirit:name>' "$component_xml" \
    | grep -E '\.(v|sv|svh|mem|dat|xdc)' \
    | sed 's|.*<spirit:name>||; s|</spirit:name>.*||' \
    | sort -u)

if (( missing_on_disk == 0 )); then
    pass "All files in component.xml exist on disk"
fi
echo

# ─── Summary ─────────────────────────────────────────────────────────
echo "============================================"
if (( errors > 0 )); then
    echo " RESULT: FAIL ($errors error(s), $warnings warning(s))"
    echo
    echo " To fix:"
    echo "   1. bash scripts/sync-ip-src.sh"
    echo "   2. vivado -mode batch -source hardware/scripts/package_accelerator_ip.tcl"
    echo "   3. bash scripts/validate-ip-repo.sh   # re-check"
else
    if (( warnings > 0 )); then
        echo " RESULT: PASS with $warnings warning(s)"
    else
        echo " RESULT: PASS (all checks clean)"
    fi
fi
echo "============================================"

exit $(( errors > 0 ? 1 : 0 ))
