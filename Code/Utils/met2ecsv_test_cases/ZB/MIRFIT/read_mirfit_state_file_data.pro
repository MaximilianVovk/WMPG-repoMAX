pro read_mirfit_state_file_data,meteor0,dir,stfl,seq,time,ts0,mir_site1,mir_site2


;program to read mirfit state file and output t,L,
mir_site1={mir_site,event:'',site:0,ts:0L,tu:0L,lat:0.0,lon:0.0,elev:0.0,th0:0.0,phi0:0.0,rot:0.0,wid:0,ht:0,offset:0.0,site_name:'',starcat:''}
;mir_site1={mir_site_data,event:'',ts:0L,tu:0L,lat:0.0,lon:0.0,elev:0.0,th0:0.0,phi0:0.0}
meteor_mir={meteor_mir_data,site:0,type:'',fr:0,id:0,cx:0.0,cy:0.0,th:0.0,phi:0.0,tau:0.0,lsp:0.0,cr:0,cg:0,cb:0,ts:0L,tu:0L,t:0.0,L:0.0,R:0.0,ht:0.0,vel:0.0,rv:0.0,rh:0.0,lat:0.0,lon:0.0}
; rx:0.0,ry:0.0,px:0,py:0,rad:0,qp:0,th_err:0.0,mag:0.0,mag_err:0.0,$
;          ,star:0,ox:0,oy:0,pos:0.0,off:0.0,sc:0,no:0}

mir_site2=mir_site1
;stop


if n_elements(stfl) eq 0 then stfl=rmd_pickfile(FILTER_IN='*.met',Title='Select mirfit state file ')

openr,1,stfl
dir=file_dirname(stfl)
;stop
       nlines=file_lines(stfl)

ts_arr=replicate(0l,nlines)
tu_arr=ts_arr
st_arr=replicate(0,nlines)

seq_arr=lonarr(nlines)
;L_arr=fltarr(nlines)
;ht_arr=fltarr(nlines)
;mag_arr=fltarr(nlines)
;mag_err=fltarr(nlines)

       result=   file_test(stfl)
       openr,inunit,stfl,/GET_LUN
n=0
i=0
k=0


       line='     '
        readf,inunit,line
        print,line
       S0 = STRSPLIT(line, ' ', /EXTRACT)

while not eof(inunit) do begin
        readf,inunit,line
        print,line
       S0 = STRSPLIT(line, ' ', /EXTRACT)
       print,s0

;stop
if s0[0] eq 'video' and s0[3] eq '1' or (s0[3] eq '4') then begin
;ev_20200801_074801A_01T.vid
;stop
mir_site1.event=s0[7]
mir_site1.ts=long(s0[29])
mir_site1.tu=long(s0[31])
mir_site1.lat=float(s0[33])
mir_site1.lon=float(s0[35])
mir_site1.elev=float(s0[37])
mir_site1.th0=float(s0[39])
mir_site1.phi0=float(s0[41])
mir_site1.offset=float(s0[where(S0 eq 'offset',count)+1])
mir_site1.rot=float(s0[where(S0 eq 'rotate',count)+1])
        readf,inunit,line
        ;print,line
       S0 = STRSPLIT(line, ' ', /EXTRACT)

endif

if (s0[0] eq 'plate' and s0[3] eq '1') or  (s0[0] eq 'plate' and s0[3] eq '4') then begin
 print,line
mir_site1.site=s0[where(S0 eq 'site' ,count)+1]
mir_site1.starcat=long(s0[where(S0 eq 'starcat' ,count)+1])
mir_site1.wid=fix(s0[where(S0 eq 'wid' ,count)+1])
mir_site1.ht=fix(s0[where(S0 eq 'ht',count)+1])
mir_site1.site_name=float(s0[where(S0 eq 'sitename',count)+1])
stop
endif



if s0[0] eq 'video' and s0[3] eq '2'  then begin
;stop
mir_site2.event=s0[7]
mir_site2.ts=long(s0[29])
mir_site2.tu=long(s0[31])
mir_site2.lat=float(s0[33])
mir_site2.lon=float(s0[35])
mir_site2.elev=float(s0[37])
mir_site2.th0=float(s0[39])
mir_site2.phi0=float(s0[41])
mir_site2.offset=float(s0[where(S0 eq 'offset',count)+1])
mir_site2.rot=float(s0[where(S0 eq 'rotate',count)+1])
        readf,inunit,line
        ;print,line
       S0 = STRSPLIT(line, ' ', /EXTRACT)

endif

if s0[0] eq 'video' and s0[3] eq '2'  then begin
 print,line
mir_site2.site=s0[where(S0 eq 'site' ,count)+1]
mir_site2.starcat=long(s0[where(S0 eq 'starcat' ,count)+1])
mir_site2.wid=fix(s0[where(S0 eq 'wid' ,count)+1])
mir_site2.ht=fix(s0[where(S0 eq 'ht',count)+1])
mir_site2.site_name=float(s0[where(S0 eq 'sitename',count)+1])
stop
endif



    while strtrim(s0[0],2) ne  'mark' and not eof(inunit) do begin

        readf,inunit,line
       S0 = STRSPLIT(line, ' ', /EXTRACT)
if s0[0] eq 'frame' then begin
    st_arr[n]=s0[7]
    ts_arr[n]=s0[9]
    tu_arr[n]=s0[11]
    seq_arr[n]=s0[13]
    n=n+1
    endif
    endwhile
;stop
for j=0,n_elements(tag_names(meteor_mir))-1 do begin
    ind=where(strupcase(S0) eq strupcase((tag_names(meteor_mir))[j]) ,count)
    ;if count gt 0 then meteor.(j)=s0[ind+1]
    if count gt 0 then begin
    if s0[5] eq 'meteor' then meteor_mir.(j)=s0[ind+1]
    endif
endfor

if count gt 0 then begin
if k eq 0 and s0[5] eq 'meteor' then begin
meteor0=meteor_mir
k=k+1
endif else begin
if s0[5] eq 'meteor' then meteor0=[meteor0,meteor_mir]
endelse
endif
i+=1
;stop

endwhile

free_lun,inunit
close,1
;stop



ind=where(ts_arr gt 0,count)
if count gt 0 then begin
ts_arr=ts_arr[ind]
tu_arr=tu_arr[ind]
st_arr=st_arr[ind]
seq_arr=seq_arr[ind]
endif
seq=0
;stop
if (size(meteor0))[0] eq 0 then return
time0=ts_arr+tu_arr/1000000.0d
time=meteor0.ts+meteor0.tu/1000000.0d - meteor0[0].ts
indt=where_a_eq_b(time0-meteor0[0].ts,time)
if indt[0] gt 0 then ts0=ts_arr[indt[0]] else ts0=0.
if indt[0] gt 0 then seq=seq_arr[indt] else seq=0

;stop

return

end