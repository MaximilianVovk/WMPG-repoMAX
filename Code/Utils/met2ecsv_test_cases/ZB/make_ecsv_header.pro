pro make_ecsv_header,mir_site,tu,iunit,dir,event


if strmid(event,22,1) eq 'A' then cat='STARS9TH_VBVRI R band'
if strmid(event,22,1) eq 'T' then cat='STARS9TH_VBVRI R band'
if strmid(event,22,1) eq 'K' then cat='STARS9TH_VBVRI R band'
if strmid(event,22,1) eq 'G' or strmid(event,22,1) eq 'F'then cat='GAIA G band'
Hp=strtrim(fix(mir_site.wid),2)
Vp=strtrim(fix(mir_site.ht),2)
;event=mir_site.event
year=strmid(event,3,4)
month=strmid(event,7,2)
day=strmid(event,9,2)
hour=strmid(event,12,2)
minute=strmid(event,14,2)
sec=strmid(event,16,2)

cam=strmid(event,20,3)
;stop
if strlen(event) gt 23 then origin=strmid(event,24,3) else origin='MET'
if strmid(cam,2,1) eq 'T' then origin='Mirfit'
lat=strtrim(string(mir_site.lat),2)
lon=strtrim(string(mir_site.lon),2)
elev=strtrim(string(mir_site.elev),2)
tu=strtrim(string(mir_site.tu/1000000.),2)
tu=strmid(string(tu),1,7)
time=year+'-'+month+'-'+day+'T'+hour+':'+minute+':'+sec+tu
print,time
;stop

obs_az=360.-(mir_site.phi0-90.)
obs_ev=90.-mir_site.th0
if strmid(event,22,1) eq 'A' then obs_rot=0.0 else obs_rot=mir_site.rot


openw,iunit,dir+'\'+event+'.ecsv'

printf,iunit,'# %ECSV 0.9'
printf,iunit,'# ---'
printf,iunit,'# datatype:'
printf,iunit,'# - {name: datetime, datatype: string}'
printf,iunit,'# - {name: ra, unit: deg, datatype: float64}'
printf,iunit,'# - {name: dec, unit: deg, datatype: float64}'
printf,iunit,'# - {name: azimuth, datatype: float64}'
printf,iunit,'# - {name: altitude, datatype: float64}'
printf,iunit,'# - {name: mag_data, datatype: float64}'
printf,iunit,'# - {name: x_image, unit: pix, datatype: float64}'
printf,iunit,'# - {name: y_image, unit: pix, datatype: float64}'
printf,iunit,'# delimiter:  '+string(39B)+','+string(39B)
printf,iunit,'# meta: !!omap'
printf,iunit,'# - {obs_latitude:',lat,'}'
printf,iunit,'# - {obs_longitude: ',lon,'}'
printf,iunit,'# - {obs_elevation: ',elev,'}'
printf,iunit,'# - {origin: '+origin+'}'
printf,iunit,'# - {camera_id: '+string(39B)+cam+string(39B)+'-'+origin+'}'
printf,iunit,'# - {cx: '+Hp+'}'
printf,iunit,'# - {cy: '+Vp+'}'
printf,iunit,'# - {photometric_band: '+cat+'}'
printf,iunit,'# - {image_file: ''',event+'.png'+'}'
printf,iunit,'# - {isodate_start_obs: '+time+'}'
printf,iunit,'# - {astrometry_number_stars: 52}'
printf,iunit,'# - {mag_label: ''mag''}'
printf,iunit,'# - {no_frags: 1}'
printf,iunit,'# - {obs_az: '+strtrim(string(obs_az,format='(f18.14)'),2)+'}'
printf,iunit,'# - {obs_ev: '+strtrim(string(obs_ev,format='(f18.14)'),2)+'}'
printf,iunit,'# - {obs_rot: '+strtrim(string(obs_rot,format='(f18.14)'),2)+'}'
;printf,iunit,'# - {obs_az: 6.87165941205465}'
;printf,iunit,'# - {obs_ev: 65.92719070293538}'
;printf,iunit,'# - {obs_rot: 71.22547630096619}'
printf,iunit,'# - {fov_horiz: 14.754961247941647}'
printf,iunit,'# - {fov_vert: 14.752316747496725}'
printf,iunit,'# schema: astropy-2.0'
printf,iunit,'datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,mag_data'
;2020-08-01T07:48:01.296434,340.933426,+72.663797,357.012733,+60.325517,   57.273,  109.864,      1659,  +7.76
;2020-08-01T07:48:01.327552,341.088126,+73.271697,357.252372,+59.727804,   36.414,  113.657,     10502,  +5.73
;2020-08-01T07:48:01.358670,341.080422,+73.918014,357.399081,+59.085811,   14.038,  115.933,     17042,  +

return

end