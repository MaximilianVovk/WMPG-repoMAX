pro met2ecsv,hp,vp,cam,dir


;program to make ecsv file from state file

;phoffset1=16.8
;phoffset2=16.8

stream='K'

close,1,2

stop
;read_state_file_data,meteor0,dir,stfl,seq,time,ts0
read_state_file_data_plus,meteor0,dir,stfl1,seq1,time1,ts1,mir_site1,mir_site2
stop
iunit=1
cam='01'+stream
event='ev_'+strmid(file_basename(dir),0,15)+'A_'+cam
make_ecsv_header,mir_site1,tu,iunit,dir,event
iunit=2
cam='02'+stream
event='ev_'+strmid(file_basename(dir),0,15)+'A_'+cam
make_ecsv_header,mir_site2,tu2,iunit,dir,event


stop

ind1=where( meteor0.site eq 1, count)
if count lt 1 then ind1=where(meteor0.site eq 4, count)
ind2=where( meteor0.site eq 2, count)
alt1=90.-meteor0[ind1].th
alt2=90.-meteor0[ind2].th
az1=(180+270-meteor0[ind1].phi) mod 360
az2=(180+270-meteor0[ind2].phi) mod 360

stop

ts1=meteor0[ind1].ts+meteor0[ind1].tu/1000000.0d
ts2=meteor0[ind2].ts+meteor0[ind2].tu/1000000.0d
;ra1=alt1
;dec1=alt1
;ra2=alt2
;dec2=alt2

;event=strmid(mir_site1.event,1,23)

form1='(a26,a1,4(f10.6,1a),2(f9.3,a1),i10.2,a1,f7.2)'


for i=0,n_elements(alt1)-1 do begin

jd1=cmsystime(ts1,/jul)
caldat,jd1,month1,day1,year1,h1,m1,s1

year1=strtrim(string(year1[i]),2)
if month1[i] lt 10 then mn1=string(month1[i],format='(i02)') else mn1=string(month1[i],format='(i2)')
if day1[i] lt 10 then dd1=string(day1[i],format='(i02)') else dd1=string(day1[i],format='(i2)')
if h1[i] lt 10 then hh1=string(h1[i],format='(i02)') else hh1=string(h1[i],format='(i2)')
if m1[i] lt 10 then mm1=string(m1[i],format='(i02)') else mm1=string(m1[i],format='(i2)')
ss1=strtrim(string(s1[i],format='(f9.6)'),2)
if fix(s1[i]) lt 10 then ss1='0'+strtrim(ss1,2)
datetime1=year1+'-'+mn1+'-'+dd1+'T'+hh1+':'+mm1+':'+ss1
cx1=meteor0[ind1[i]].cx
cy1=meteor0[ind1[i]].cy
lsp1=10^(-0.4*(meteor0[ind1[i]].lsp))
mag1= meteor0[ind1[i]].mag ;lsp-mir_site1.offset

hor2eq, alt1[i], az1[i], jd1[i], ra1, dec1, ha, lat=lat, lon=lon, WS=WS,verbose=verbose, precess_=precess_, nutate_=nutate_, altitude=elevation

printf,1,datetime1,',',ra1,',',dec1,',',az1[i],',',alt1[i],',',cx1,',',cy1,',',lsp1,',',mag1,format=form1

endfor


for i=0,n_elements(alt2)-1 do begin
jd2=cmsystime(ts2,/jul)
caldat,jd2,month2,day2,year2,h2,m2,s2
year2=strtrim(string(year2[i]),2)
if month2[i] lt 10 then mn2=string(month2[i],format='(i02)') else mn2=string(month2[i],format='(i2)')
if day2[i] lt 10 then dd2=string(day2[i],format='(i02)') else dd2=string(day2[i],format='(i2)')
if h2[i] lt 10 then hh2=string(h2[i],format='(i02)') else hh2=string(h2[i],format='(i2)')
if m2[i] lt 10 then mm2=string(m2[i],format='(i02)') else mm2=string(m2[i],format='(i2)')
ss2=strtrim(string(s2[i],format='(f9.6)'),2)
if fix(s2[i]) lt 10 then ss2='0'+strtrim(ss2,2)
datetime2=year2+'-'+mn2+'-'+dd2+'T'+hh2+':'+mm2+':'+ss2
cx2=meteor0[ind2[i]].cx
cy2=meteor0[ind2[i]].cy
lsp2=  10^(-0.4*(meteor0[ind2[i]].lsp))
mag2= meteor0[ind2[i]].mag ;lsp+mir_site2.offset

hor2eq, alt2[i], az2[i], jd2[i], ra2, dec2, ha, lat=lat, lon=lon, WS=WS,verbose=verbose, precess_=precess_, nutate_=nutate_, altitude=elevation

printf,2,datetime2,',',ra2,',',dec2,',',az2[i],',',alt2[i],',',cx2,',',cy2,',',lsp2,',',mag2,format=form1


endfor

close,1,2

stop


return
end

