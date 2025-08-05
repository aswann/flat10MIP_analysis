#! /bin/tcsh

module load nco

# Script which makes means and time series of CESM model output
# ALSS 2024.05.10 

# This is the name of the cases we'll make means/time series of

# for flat10 runs:
#set caselist_online = (b.e21.B1850.f09_g17.FLAT10-cdr.001 b.e21.B1850.f09_g17.FLAT10-esm.001 b.e21.B1850.f09_g17.FLAT10-zec.001) #b.e21.B1850.f09_g17.FLAT10ctrl-esm.001 
set caselist_online = (NorESM2-LM_flat10_zec) # _1950-01-1959-12)

# List the variables to rename
set rename_vars = (tas rh cVeg cSoil cLitter gpp nbp)

# filenames look like this: cVeg_NorESM2-LM_flat10_zec_1950-01-1959-12.nc

# flag for renaming variables
set do_rename = 1  

# Loop over each online case:
foreach casename ($caselist_online)

	echo "casename = " $casename
	echo "user = " $user
	set workdir = /glade/campaign/cgd/tss/people/aswann/flat10/NorESM2-LM/wrongformat

    echo "workdir = " $workdir

	#set outputdir = /glade/derecho/scratch/$user/flat10 
	set outputdir = /glade/campaign/cgd/tss/people/$user/flat10/NorESM2-LM/wrongformat

    echo "outputdir = " $outputdir


	if ($do_rename == 1) then
		foreach renamevar ($rename_vars)
			echo "renaming " $renamevar
            if ($renamevar == "nbp") then
                set CMORvar = NBP 
                set mfilename = Emon
            else if ($renamevar == "cLitter") then
                set CMORvar = TOTLITC 
                set mfilename = Emon
            else if ($renamevar == "cSoil") then
                set CMORvar = TOTSOMC
                set mfilename = Emon
            else if ($renamevar == "cVeg") then
                set CMORvar = TOTVEGC
                set mfilename = Emon
            else if ($renamevar == "gpp") then
                set CMORvar = GPP 
                set mfilename = Emon
            else if ($renamevar == "rh") then
                set CMORvar = HR 
                set mfilename = Emon
            else if ($renamevar == "tas") then
                set CMORvar = TREFHT
                set mfilename = Amon
            endif

			echo "converting name to " $renamevar " from " $CMORvar
			# use nco to rename the variable and convert units if necessary
			#cp $workdir/${renamevar}_$casename.nc ${finalfilename}    #make a copy with a different name 

            set finalfilename = "${workdir}/${renamevar}_${mfilename}_${casename}_r1i1p1f1_gn_195001-195912.nc" #cSoil_Emon_NorESM2-LM_flat10_zec_r1i1p1f1_gn_196001-196912.nc
            cp "${workdir}/${renamevar}_${casename}_1950-01-1959-12.nc" ${finalfilename}   
            #echo "copied correctly"
            ncrename -v $CMORvar,$renamevar ${finalfilename} # rename the variable inside the file

            #cp $ts_dir/$casename.$mfilename.ts.allyears.$renamevar.nc $ts_dir/$casename.$mfilename.ts.allyears.$CMORvar.nc
            #ncrename -v $renamevar,$CMORvar $ts_dir/$casename.$mfilename.ts.allyears.$CMORvar.nc

            # adjust the units if necessary
            # stocks need to be converted from gC/m2 to kgC/m2
            # fluxes need to be converted from gC/m2/s to kgC/m2/s

			if ($renamevar == "gpp") then
                echo "converting gpp unit"
				# convert units by mulitplying by a constant
				ncap2 -O -s 'gpp*=0.001'  ${finalfilename}    out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,gpp,o,c,kgC/m2/s out.nc ${finalfilename}   
			endif

            if ($renamevar == "nbp") then
				# convert units by mulitplying by a constant
				ncap2 -O -s 'nbp*=0.001'  ${finalfilename} out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,nbp,o,c,kgC/m2/s out.nc ${finalfilename}
			endif

			if ($renamevar == "rh") then
				# convert units by mulitplying by a constant
				ncap2 -O -s 'rh*=0.001'  ${finalfilename}    out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,rh,o,c,kgC/m2/s out.nc ${finalfilename}   
			endif

			if ($renamevar == "cVeg") then
				# convert units by mulitplying by a constant
				ncap2 -O -s 'cVeg*=0.001'  ${finalfilename}    out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,cVeg,o,c,kgC/m2 out.nc ${finalfilename}   
			endif

			if ($renamevar == "cSoil") then
				# convert units by mulitplying by a constant
				ncap2 -O -s 'cSoil*=0.001'  ${finalfilename}    out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,cSoil,o,c,kgC/m2 out.nc ${finalfilename}   
			endif

			if ($renamevar == "cLitter") then
				# convert units by mulitplying by a constant
				ncap2 -O -s 'cLitter*=0.001'  ${finalfilename}    out.nc
				# rename the unit attribute to match the new unit
				ncatted -O -a units,cLitter,o,c,kgC/m2 out.nc ${finalfilename}   
			endif

		end #foreach renamevar
    end #if dorename
end # foreach casename 