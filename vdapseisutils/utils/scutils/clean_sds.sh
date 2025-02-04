#!/bin/bash
#
# THIS PROGRAM REQUIRES THE INSTLLATION OF SEISCOMP
#

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 archive"
    exit 1
fi

archive="$1"

# Check for unordered records
echo "Checking for unordered records..."
#/home/sysop/seiscomp/bin/scart --check . > check_results.txt 2>&1
/home/sysop/seiscomp/bin/scart --check $archive 2>&1 | tee check_results.txt
echo ""

# Extract file paths with issues
files_with_issues=$(grep "has an issue" check_results.txt | awk '{print $1}')
echo "Files with issues:"
#echo $files_with_issues
printf "%s\n" $files_with_issues
echo ""

# Loop through each file and sort it
echo "Sorting files that have an issue..."
for filepath in $files_with_issues; do
    echo "Sorting file: $filepath"
    /home/sysop/seiscomp/bin/scmssort -vuiE "$filepath" > tmp.mseed && mv tmp.mseed "$filepath"
    echo ""
done

# Check again for unordered records
echo "Re-checking for unordered records..."
/home/sysop/seiscomp/bin/scart --check $archive

