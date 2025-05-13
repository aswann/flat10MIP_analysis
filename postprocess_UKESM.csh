cd /glade/campaign/cgd/tss/people/aswann/flat10/

# /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/FLAT10

#cp -r UKESM1.2 unprocessed_UKESM1.2

cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/

mv FLAT10 flat10
mv FLAT10CDR  flat10-cdr
mv FLAT10ZEC flat10-zec

#mkdir flat10
#mkdir flat10-zec
#mkdir flat10-cdr

cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10

mv dg403_flat10_soil_carbon_kgC_m2.nc dg403_flat10_cSoil.nc
mv dg403_flat10_vegetation_carbon_kgC_m2.nc dg403_flat10_cVeg.nc
mv dg403_flat10_air_to_sea_CO2_flux_kgCO2_m2_s.nc dg403_flat10_fgco2.nc
mv dg403_flat10_gpp_kgC_m2_yr.nc dg403_flat10_gpp.nc
mv dg403_flat10_npp_after_n_limitation.nc dg403_flat10_npp.nc
mv dg403_flat10_precipitation_kg_m2_s.nc dg403_flat10_pr.nc
mv dg403_flat10_1p5m_temperature_K.nc dg403_flat10_tas.nc
mv dg403_flat10_soil_respiration_kgC_m2_yr.nc dg403_flat10_rh.nc

ncatted -a _FillValue,,o,f,NaN in.nc
ncatted -a _FillValue,,o,f,NaN test.nc


cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10-cdr

mv dh493_flat10cdr_soil_carbon_kgC_m2.nc dg493_flat10-cdr_cSoil.nc
mv dh493_flat10cdr_vegetation_carbon_kgC_m2.nc dg493_flat10-cdr_cVeg.nc
mv dh493_flat10cdr_air_to_sea_CO2_flux_kgCO2_m2_s.nc dg493_flat10-cdr_fgco2.nc
mv dh493_flat10cdr_gpp_kgC_m2_yr.nc dg493_flat10-cdr_gpp.nc
mv dh493_flat10cdr_npp_after_n_limitation.nc dg493_flat10-cdr_npp.nc
mv dh493_flat10cdr_precipitation_kg_m2_s.nc dg493_flat10-cdr_pr.nc
mv dh493_flat10cdr_1p5m_temperature_K.nc dg493_flat10-cdr_tas.nc
mv dg493_flat10cdr_soil_respiration_kgC_m2_yr.nc dg493_flat10-cdr_rh.nc


cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10-zec

mv dh492_flat10zec_soil_carbon_kgC_m2.nc dg492_flat10-zec_cSoil.nc
mv dh492_flat10zec_vegetation_carbon_kgC_m2.nc dg492_flat10-zec_cVeg.nc
mv dh492_flat10zec_air_to_sea_CO2_flux_kgCO2_m2_s.nc dg492_flat10-zec_fgco2.nc
mv dh492_flat10zec_gpp_kgC_m2_yr.nc dg492_flat10-zec_gpp.nc
mv dh492_flat10zec_npp_after_n_limitation.nc dg492_flat10-zec_npp.nc
mv dh492_flat10zec_precipitation_kg_m2_s.nc dg492_flat10-zec_pr.nc
mv dh492_flat10zec_1p5m_temperature_K.nc dg492_flat10-zec_tas.nc
mv dg492_flat10zec_soil_respiration_kgC_m2_yr.nc dg492-zec_flat10_rh.nc

