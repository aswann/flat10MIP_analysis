
module load cdo 

# ncbo --op_typ=add 1.nc 2.nc 3.nc’, ‘ncadd 1.nc 2.nc 3.nc
# cdo setname,rsus -sub rsds.nc rsns.nc rsus.nc

cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10/
cdo setname,nbp -sub dg403_flat10_npp.nc dg403_flat10_rh.nc dg403_flat10_nbp.nc

cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10-zec/
cp dh492_flat10zec_soil_respiration_kgC_m2_yr.nc dg492_flat10-zec_rh.nc
cdo setname,nbp -sub dg492_flat10-zec_npp.nc dg492_flat10-zec_rh.nc dg492_flat10-zec_nbp.nc

cd /glade/campaign/cgd/tss/people/aswann/flat10/UKESM1.2/flat10-cdr/

cp dh493_flat10cdr_soil_respiration_kgC_m2_yr.nc dg493_flat10-cdr_rh.nc
cdo setname,nbp -sub dg493_flat10-cdr_npp.nc dg493_flat10-cdr_rh.nc dg493_flat10-cdr_nbp.nc


