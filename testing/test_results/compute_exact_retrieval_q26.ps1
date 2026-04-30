$rels=@{}
$rels['Q1']=@(1,2,3,4,6,7,9)
$rels['Q2']=@(1,2,3,4,5,6,7)
$rels['Q3']=@(1,2,3,4,5,6,7,8,10)
$rels['Q4']=@(3,5,6,7,8,9,10)
$rels['Q5']=@(2,4,6,8)
$rels['Q6']=@(1,3,4,5,6,7,8,9,10)
$rels['Q7']=@(1,2,3,4,5,8,9,10)
$rels['Q8']=@(1,2,3,4,9)
$rels['Q9']=@(1,2,3,4,5,6,9)
$rels['Q10']=@(1,2,3,4,5,6,7,8)
$rels['Q11']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q12']=@(1,2,3,4,5,7,8)
$rels['Q13']=@(1,2,3,4,5,6,7,8,10)
$rels['Q14']=@(1,2,3,4,5,6,9,10)
$rels['Q15']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q16']=@(1,2,3,5,6,7,8,9)
$rels['Q17']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q18']=@(1,2,3,4,5,6,7,8,10)
$rels['Q19']=@(2,3,5,8,9,10)
$rels['Q20']=@(1,2,3,4,5,6,7,8,10)
$rels['Q21']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q22']=@(1,2,3,4,5,6,7,8,10)
$rels['Q23']=@()
$rels['Q24']=@(1,3,4,6,7,9)
$rels['Q25']=@(1,2,3,4,5)
$rels['Q26']=@(1,2,3,4,5,6,7,9)
$rels['Q27']=@(1,2,3,4,5,6,7,8,10)
$rels['Q28']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q29']=@(2,3,4,9)
$rels['Q30']=@(1)
$rels['Q31']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q32']=@(1,2,3,4,5,6,7,8)
$rels['Q33']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q34']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q35']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q36']=@(1,2,3,4,5,6,7,9,10)
$rels['Q37']=@(2,7,8)
$rels['Q38']=@(1,2,3,4,7,9,10)
$rels['Q39']=@(7)
$rels['Q40']=@(1,2,3,4,5,6,7,8,9,10)
$rels['Q41']=@(1,2,3,4,5,6,7,8,9)
$rels['Q42']=@(2,7)
$rels['Q43']=@(1,2,3,4,5,6,7,8,9)
$rels['Q44']=@(1,3,4,6,7,9,10)
$rels['Q45']=@(2,7)

function Get-AP10([int[]]$ranks){
  if($ranks.Count -eq 0){ return 0.0 }
  $sorted=$ranks|Sort-Object
  $hits=0
  $sumP=0.0
  foreach($k in $sorted){
    if($k -le 10){
      $hits += 1
      $sumP += ($hits / $k)
    }
  }
  if($hits -eq 0){ return 0.0 }
  return $sumP / $hits
}

function Get-nDCG10([int[]]$ranks){
  if($ranks.Count -eq 0){ return 0.0 }
  $sorted=$ranks|Sort-Object
  $dcg=0.0
  foreach($k in $sorted){
    if($k -le 10){
      $dcg += 1.0 / ([math]::Log($k+1,2))
    }
  }
  $m=($sorted|Where-Object{$_ -le 10}).Count
  if($m -eq 0){ return 0.0 }
  $idcg=0.0
  for($i=1; $i -le $m; $i++){
    $idcg += 1.0 / ([math]::Log($i+1,2))
  }
  return $dcg / $idcg
}

$rows=@()
foreach($q in 1..45){
  $id='Q'+$q
  $r=$rels[$id]
  $hit= if($r.Count -gt 0){1}else{0}
  $mrr= if($r.Count -gt 0){ 1.0 / ((($r|Sort-Object)[0])) } else {0.0}
  $ap=Get-AP10 $r
  $ndcg=Get-nDCG10 $r
  $rows += [pscustomobject]@{id=$id; relCount=$r.Count; hit=$hit; mrr=$mrr; ap10=$ap; ndcg10=$ndcg}
}

$summary=[pscustomobject]@{
  Hit10=[math]::Round((($rows|Measure-Object hit -Average).Average),3)
  MRR10=[math]::Round((($rows|Measure-Object mrr -Average).Average),3)
  MAP10=[math]::Round((($rows|Measure-Object ap10 -Average).Average),3)
  nDCG10=[math]::Round((($rows|Measure-Object ndcg10 -Average).Average),3)
}

$summary | Format-List
''
'Miss queries:'
($rows | Where-Object {$_.hit -eq 0} | Select-Object -ExpandProperty id) -join ', '
''
'Rows with low AP (<0.4):'
$rows | Where-Object {$_.ap10 -lt 0.4} | Select-Object id,relCount,mrr,ap10,ndcg10 | Format-Table -AutoSize
