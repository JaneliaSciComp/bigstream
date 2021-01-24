#!/bin/bash


function get_job_dependency {
  unset dependency_string
  IFS="," read -a dependencies <<< "${1?}"
  for dep in "${dependencies[@]}"; do
      bjobs_lines=`bjobs -J "$dep"`
      jobids=`echo "$bjobs_lines" | cut -f 1 -d' ' | tail -n +2 | uniq`
      for jobid in ${jobids[@]}; do
        dependency_string="${dependency_string}ended($jobid)&&"
      done
  done
  dependency_string=${dependency_string::-2}
  echo $dependency_string
}


function submit {
    name="${1?}";       shift
    dependency="${1?}"; shift
    cores="${1?}";      shift
    execute="$@"

    [[ -z "$dependency" ]] || dependency=$(get_job_dependency "$dependency")
    [[ -z "$dependency" ]] || dependency="-w $dependency"

    bsub -J $name \
         -n $cores \
         -o ${logdir}/${name}.o \
         -e ${logdir}/${name}.e \
         -P $BILLING \
         $dependency \
         "$execute"
}


function initialize_environment {
  logdir="${outdir}/logs";    mkdir -p $logdir
  affdir="${outdir}/aff";     mkdir -p $affdir
  tiledir="${outdir}/tiles";  mkdir -p $tiledir
  BILLING='multifish'
  PYTHON='/groups/scicompsoft/home/fleishmang/bin/miniconda3/envs/bigstream_test/bin/python'
  SCRIPTS='/nrs/scicompsoft/goinac/multifish/ex1/greg_test/bigstream/bigstream'
  CUT_TILES="$PYTHON ${SCRIPTS}/cut_tiles.py"
  SPOTS="$PYTHON ${SCRIPTS}/spots.py"
  RANSAC="$PYTHON ${SCRIPTS}/ransac.py"
  INTERPOLATE_AFFINES="$PYTHON ${SCRIPTS}/interpolate_affines.py"
  DEFORM="$PYTHON ${SCRIPTS}/deform.py"
  STITCH="$PYTHON ${SCRIPTS}/stitch_and_write.py"
  APPLY_TRANSFORM="$PYTHON ${SCRIPTS}/apply_transform_n5.py"
}


fixed=${1?}; shift
moving=${1?}; shift
outdir=${1?}; shift
channel=${1?}; shift
aff_scale=${1?}; shift
def_scale=${1?}; shift
xy_stride=${1?}; shift
z_stride=${1?}; shift
spots_cc_radius=${1?}; shift
spots_spot_number=${1?}; shift
ransac_cc_cutoff=${1?}; shift
ransac_dist_threshold=${1?}; shift
deform_iterations=${1?}; shift
deform_auto_mask=${1?}; shift

xy_overlap=$(( $xy_stride / 8 ))
z_overlap=$(( $z_stride / 8 ))


initialize_environment

submit "cut_tiles" '' 1 \
$CUT_TILES $fixed /${channel}/${def_scale} $tiledir $xy_stride $xy_overlap $z_stride $z_overlap

submit "coarse_spots" '' 1 \
$SPOTS coarse $fixed /${channel}/${aff_scale} ${affdir}/fixed_spots.pkl $spots_cc_radius $spots_spot_number

submit "coarse_spots" '' 1 \
$SPOTS coarse $moving /${channel}/${aff_scale} ${affdir}/moving_spots.pkl $spots_cc_radius $spots_spot_number

submit "coarse_ransac" "coarse_spots" 1 \
$RANSAC ${affdir}/fixed_spots.pkl ${affdir}/moving_spots.pkl \
        ${affdir}/ransac_affine.mat $ransac_cc_cutoff $ransac_dist_threshold

submit "apply_affine_small" "coarse_ransac" 1 \
$APPLY_TRANSFORM $fixed /${channel}/${aff_scale} $moving /${channel}/${aff_scale} \
                 ${affdir}/ransac_affine.mat ${affdir}/ransac_affine

submit "apply_affine_big" "coarse_ransac" 8 \
$APPLY_TRANSFORM $fixed /${channel}/${def_scale} $moving /${channel}/${def_scale} \
                 ${affdir}/ransac_affine.mat ${affdir}/ransac_affine


while [[ ! -f ${tiledir}/0/coords.txt ]]; do
    sleep 1
done
sleep 5


for tile in $( ls -d ${tiledir}/*[0-9] ); do

  tile_num=`basename $tile`

  submit "spots${tile_num}" '' 1 \
  $SPOTS ${tile}/coords.txt $fixed /${channel}/${aff_scale} ${tile}/fixed_spots.pkl $spots_cc_radius $spots_spot_number

  submit "spots${tile_num}" "apply_affine_small" 1 \
  $SPOTS ${tile}/coords.txt ${affdir}/ransac_affine /${channel}/${aff_scale} ${tile}/moving_spots.pkl $spots_cc_radius $spots_spot_number

  submit "ransac${tile_num}" "spots${tile_num}" 1 \
  $RANSAC ${tile}/fixed_spots.pkl ${tile}/moving_spots.pkl ${tile}/ransac_affine.mat \
          $ransac_cc_cutoff $ransac_dist_threshold
done

while [[ ! -f ${tiledir}/0/ransac_affine.mat ]]; do
    sleep 1
done
sleep 5

submit "interpolate_affines" 'ransac*' 1 \
$INTERPOLATE_AFFINES $tiledir

for tile in $( ls -d ${tiledir}/*[0-9] ); do
  tile_num=`basename $tile`
  submit "deform${tile_num}" "interpolate_affines,apply_affine_big" 1 \
  $DEFORM $fixed /${channel}/${def_scale} ${affdir}/ransac_affine /${channel}/${def_scale} \
          ${tile}/coords.txt ${tile}/warp.nrrd \
          ${tile}/ransac_affine.mat ${tile}/final_lcc.nrrd \
          ${tile}/invwarp.nrrd $deform_iterations $deform_auto_mask
done

for tile in $( ls -d ${tiledir}/*[0-9] ); do
  tile_num=`basename $tile`
  submit "stitch${tile_num}" 'deform*' 2 \
  $STITCH $tile $xy_overlap $z_overlap $fixed /${channel}/${def_scale} ${affdir}/ransac_affine.mat \
          ${outdir}/transform ${outdir}/invtransform /${def_scale}
done

submit "apply_transform" 'stitch*' 12 \
$APPLY_TRANSFORM $fixed /${channel}/${def_scale} $moving /${channel}/${def_scale} \
                 ${outdir}/transform ${outdir}/warped

