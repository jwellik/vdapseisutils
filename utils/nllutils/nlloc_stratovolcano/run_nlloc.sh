echo "============"
echo "Script to run sample locations for NonLinLoc - Non-Global mode"
echo "see http://alomax.net/nlloc"
echo "Modified by Jay Wellik to run at Copahue volcano"
echo
echo "IMPORTANT:  Requires:"
echo "   1. NonLinLoc - the command \"NLLoc\" must be on your path"
echo "   2. Java - the command \"java\" must be on your path"
echo "   3. GMT - for postscript visualization, GMT must be installed and GMT tool commands must be on your path"
echo "   #4. SeismicityViewer must be installed and on your java classpath - see: http://alomax.net/seismicity"
echo "   #5. The environment variable PS_VIEWER is set in your shell or in this script"
echo

# PS_VIEWER=ghostview    # Linux, MacOSX
# alias ghostview="open -a Preview"    # MacOSX
# setGMT4    # make sure GMT4 and not GMT5 commands are on path

echo "::: RUNNING NONLINLOC :::"
echo "Removing gmt/* loc/* model/* time/*"
rm -r gmt/* loc/* model/* time/*
echo

echo "Generate the model grid"
Vel2Grid run/nlloc_stratovolcano.in
echo

echo "Create and visualize the model grid"
Grid2GMT run/nlloc_stratovolcano.in model/layer.P.mod gmt/ V G 1 0 1 401
#${PS_VIEWER} gmt/layer.P.mod.VG.ps &
echo

echo "Generate and view the travel-time and take-off angle grids "
Grid2Time run/nlloc_stratovolcano.in
#echo
#echo "Visualize P travel-time grid"
#Grid2GMT run/nlloc_stratovolcano.in time/layer.P.AK_RC01_--.time gmt/ V G 0 0 0 401
#${PS_VIEWER} gmt/layer.P.AK_RC01_--.time.VG.ps &
#echo "Visualize P take-off angles grid"
#Grid2GMT run/nlloc_stratovolcano.in time/layer.P.AK_RC01_--.angle gmt/ V G 0 0 0 401
#${PS_VIEWER} gmt/layer.P.AK_RC01_--.angle.VG.ps &
echo

echo "Generate some synthetic arrival times "
Time2EQ run/nlloc_stratovolcano.in
more obs_synth/synth.obs
echo

echo "Do the event Location "
NLLoc run/nlloc_stratovolcano.in

echo "Plot the combined locations with GMT"
LocSum loc/volcano.sum.grid0.loc 1 loc/volcano "loc/volcano.*.*.grid0.loc"
Grid2GMT run/nlloc_stratovolcano.in loc/volcano gmt/ L E100

echo "Plot each location w GMT"
Grid2GMT run/nlloc_stratovolcano.in "loc/volcano.*.*.grid0.loc" gmt/ L E111
Grid2GMT run/nlloc_stratovolcano.in "loc/volcano.*.*.grid0.loc" gmt/ L S
#${PS_VIEWER} gmt/volcano.LE_101.ps &
echo

#echo "Visualise the location with Seismicity Viewer (you must have installed Seismicity Viewer, see Seismicity Viewer software guide) "
#java net.alomax.seismicity.Seismicity loc/alaska.*.*.grid0.loc.hyp

echo "Done."
