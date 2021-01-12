#!/bin/bash


# the fixed n5 image
fixed="/nrs/multifish/Yuhan/LHA3/R3_LHA3/export.n5"
# the moving n5 image
moving="/nrs/multifish/Yuhan/LHA3/R5_LHA3/export.n5"
# the folder where you'd like all outputs to be written
outdir="/groups/multifish/multifish/fleishmang/alignments/LHA_R5_TO_R3"


# the channel used to drive registration
channel="c2"
# the scale level for affine alignments
aff_scale="s3"
# the scale level for deformable alignments
def_scale="s2"
# the number of voxels along x/y for registration tiling, must be power of 2
xy_stride=256
# the number of voxels along z for registration tiling, must be power of 2
z_stride=256


# spots params
spots_cc_radius="8"
spots_spot_number="2000"
# ransac params
ransac_cc_cutoff="0.9"
ransac_dist_threshold="2.5"
# deformation parameters
deform_iterations="500x200x25x1"
auto_mask="0"


# DO NOT EDIT BELOW THIS LINE
big_stream='/groups/multifish/multifish/fleishmang/stream/stream.sh'
bash "$big_stream" "$fixed" "$moving" "$outdir" "$channel" \
     "$aff_scale" "$def_scale" "$xy_stride" "$z_stride" \
     "$spots_cc_radius" "$spots_spot_number" \
     "$ransac_cc_cutoff" "$ransac_dist_threshold" \
     "$deform_iterations" "$auto_mask"

