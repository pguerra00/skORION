import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

if (!process.argv[2]) {
  console.error('Usage: node make_workbook.mjs /path/to/Analyses/my_experiment');
  process.exit(2);
}
const root=path.resolve(process.argv[2]);
const previewDir=path.resolve(process.env.FOCI_PREVIEW_DIR ?? path.join(root,'Methods','Workbook_Previews'));
const data=JSON.parse(await fs.readFile(path.join(root,'Methods','workbook_data.json'),'utf8'));
const wb=Workbook.create();
const priority={
  Image_Summary:['image_id','treatment_group','nuclei_analyzed','C2_total_foci','C2_mean_foci_per_nucleus','C3_total_foci','C3_mean_foci_per_nucleus'],
  Cell_Summary:['image_id','treatment_group','nucleus_id','nuclear_area_um2','C2_foci_count','C3_foci_count','C2_foci_density_per_um2','C3_foci_density_per_um2'],
  Focus_Data:['focus_id','image_id','treatment_group','nucleus_id','channel','focus_area_um2','raw_mean_adu','raw_integrated_adu_px'],
  QC_Summary:['image_id','treatment_group','channel','number_usable_nuclei','number_excluded_nuclei','background_median_adu','signal_to_background_ratio','manual_inspection_warnings'],
  Analysis_Parameters:['parameter','value'],
  Treatment_Mapping:['image_id','treatment_group','replicate_id','n_channels','width_px','height_px','pixel_size_x_um','pixel_size_y_um'],
  Data_Dictionary:['table','column','data_type','definition']
};
const validation={sheets:[],authoring_library:'@oai/artifact-tool',calculation_model:'Static measurements computed from original TIFFs by the saved Python pipeline. No spreadsheet formulas required.'};
await fs.mkdir(previewDir,{recursive:true});

function formatCode(k){
  if (/pixel_size/.test(k))return '0.000000';
  if (/fraction_occupied|relative_radial_position|eccentricity|circularity|solidity|aspect_ratio|ratio|density/.test(k))return '0.000';
  if (/count|total_foci|nuclei_analyzed|nucleus_label|focus_label|channel$|n_channels|_pixels$|_area_px$|biological_replicates|number_|width_px|height_px|bit_depth/.test(k))return '#,##0';
  if (/percent|pct/.test(k))return '0.0';
  return '#,##0.00';
}

for(const [name,table] of Object.entries(data)){
  const sheet=wb.worksheets.add(name);sheet.showGridLines=false;
  const start=priority[name] ?? [];
  const late=['image_name','source_path','sha256','qc_flags','C2_qc_flags','C3_qc_flags'];
  const cols=[...start,...table.columns.filter(c=>!start.includes(c)&&!late.includes(c)),...late.filter(c=>table.columns.includes(c)&&!start.includes(c))];
  const positions=cols.map(c=>table.columns.indexOf(c));
  const rows=table.rows.map(r=>positions.map(i=>r[i]));
  const matrix=[cols,...rows];
  const range=sheet.getRangeByIndexes(0,0,matrix.length,cols.length);
  range.values=matrix;
  range.format.font={name:'Arial',size:10,color:'#243746'};
  range.format.rowHeight=30;
  range.format.verticalAlignment='center';
  range.format.columnWidth=24;
  range.format.wrapText=true;
  for(let j=0;j<cols.length;j++){
    const k=cols[j];const rr=sheet.getRangeByIndexes(0,j,matrix.length,1);
    let width=24;
    if(/^(image_id|channel|nucleus_label|focus_label|marker|included|data_type)$/.test(k))width=14;
    if(/^(treatment_group|nucleus_id|focus_id|replicate_id)$/.test(k))width=22;
    if(k==='image_name')width=90;
    if(k==='source_path')width=100;
    if(k==='sha256')width=74;
    if(k==='parameter'||k==='column')width=62;
    if(k==='value')width=115;
    if(k==='definition')width=115;
    if(/warning|qc_flags/.test(k))width=100;
    rr.format.columnWidth=width;
    if(rows.length){
      const body=sheet.getRangeByIndexes(1,j,rows.length,1);
      const isNumber=rows.some(r=>typeof r[j]==='number');
      body.format.horizontalAlignment=isNumber?'right':'left';
      if(isNumber)body.setNumberFormat(formatCode(k));
    }
  }
  if(name==='Analysis_Parameters'||name==='Data_Dictionary'){
    for(let i=0;i<rows.length;i++){
      const chars=Math.max(...rows[i].map(x=>String(x??'').length));
      sheet.getRangeByIndexes(i+1,0,1,cols.length).format.rowHeight=Math.max(30,15*Math.ceil(chars/92)+9);
    }
  }
  if(name==='Image_Summary'||name==='QC_Summary')range.format.rowHeight=75;
  const head=sheet.getRangeByIndexes(0,0,1,cols.length);
  head.format.fill='#243C50';head.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  head.format.rowHeight=76;head.format.horizontalAlignment='center';
  const excelTable=sheet.tables.add(range,true,name.replaceAll('_','')+'Table');
  excelTable.style='TableStyleMedium2';
  excelTable.showFilterButton=true;
  sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(1);
  if(name==='Image_Summary')sheet.tabColor='#243C50';
  else if(name==='Analysis_Parameters')sheet.tabColor='#6E8295';
  validation.sheets.push({name,rows:rows.length,columns:cols.length,first_columns:cols.slice(0,8)});
  console.log(`${name}: ${rows.length} rows, ${cols.length} columns`);
}
wb.recalculate();
const inspect=await wb.inspect({kind:'table',range:'Image_Summary!A1:G4',include:'values,formulas',tableMaxRows:4,tableMaxCols:7,maxChars:4000});
console.log(inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500});
validation.error_scan=errors.ndjson;
await fs.mkdir(path.join(root,'Results'),{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(root,'Results','Foci_Quantification_Results.xlsx'));
for(const s of validation.sheets){
  const last=s.name==='Analysis_Parameters'?'B':s.name==='Data_Dictionary'?'D':'H';
  const preview=await wb.render({sheetName:s.name,range:`A1:${last}${Math.min(s.rows+1,5)}`,scale:1.3,format:'png'});
  await fs.writeFile(path.join(previewDir,`${s.name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
validation.xlsx_exported=true;
validation.sheet_previews_rendered=validation.sheets.map(s=>s.name);
await fs.writeFile(path.join(root,'Methods','Workbook_Validation.json'),JSON.stringify(validation,null,2));
console.log('Workbook exported; all seven sheet previews rendered.');
