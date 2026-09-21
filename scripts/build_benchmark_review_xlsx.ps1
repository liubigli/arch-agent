param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'

function Read-JsonUtf8([string]$Path) {
    return [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) | ConvertFrom-Json
}

function Json-Compact($Value) {
    if ($null -eq $Value) { return '' }
    return ($Value | ConvertTo-Json -Depth 20 -Compress)
}

$evaluationFiles = Get-ChildItem -LiteralPath $InputDir -Filter 'benchmark_evaluation_*.json' | Sort-Object Name
if ($evaluationFiles.Count -ne 5) {
    throw "Expected 5 evaluation files, found $($evaluationFiles.Count)."
}

$detail = New-Object System.Collections.Generic.List[object]
$summary = New-Object System.Collections.Generic.List[object]

foreach ($evaluationFile in $evaluationFiles) {
    $evaluation = Read-JsonUtf8 $evaluationFile.FullName
    $safeModel = ($evaluation.model -replace '[:.]', '_') -replace '-', '_'
    $rawFile = Get-ChildItem -LiteralPath $InputDir -Filter "benchmark_raw_*${safeModel}*.json" | Select-Object -First 1
    $manualFile = Get-ChildItem -LiteralPath $InputDir -Filter "benchmark_manual_review_*${safeModel}*.json" | Select-Object -First 1
    if (-not $rawFile) { throw "Raw file not found for $($evaluation.model)." }

    $raw = Read-JsonUtf8 $rawFile.FullName
    $manualIds = @{}
    if ($manualFile) {
        $manual = Read-JsonUtf8 $manualFile.FullName
        foreach ($record in $manual.records) { $manualIds[[string]$record.question_id] = $true }
    }
    $evalById = @{}
    foreach ($record in $evaluation.records) { $evalById[[string]$record.question_id] = $record }

    foreach ($record in $raw.records) {
        $ev = $evalById[[string]$record.question_id]
        $toolNames = @($record.tool_calls | ForEach-Object { $_.name })
        $toolArgs = @($record.tool_calls | ForEach-Object { Json-Compact $_.args })
        $toolOutputs = @($record.tool_calls | ForEach-Object { $_.output })
        $autoApproved = (
            $ev.reliability -eq 'grounded' -and
            $ev.tool_selection_status -in @('optimal', 'acceptable') -and
            $ev.language_ok -ne $false -and
            [string]::IsNullOrWhiteSpace([string]$ev.error)
        )
        $detail.Add([pscustomobject]@{
            model = $evaluation.model
            think = 'true'
            question_id = [int]$record.question_id
            question = [string]$record.question
            expected_language = [string]$record.expected_language
            final_answer = [string]$record.final_answer
            tool_calls = ($toolNames -join '; ')
            tool_args = ($toolArgs -join "`n")
            tool_outputs = ($toolOutputs -join "`n---`n")
            tool_call_count = $toolNames.Count
            expected_tools = (@($ev.expected_tools) -join '; ')
            acceptable_tools = (@($ev.acceptable_tools) -join '; ')
            routing_status = [string]$ev.tool_selection_status
            routing_issue = [string]$ev.tool_selection_issue
            automatic_validation = if ($autoApproved) { 'APPROVATA' } else { 'DA VERIFICARE' }
            reliability = [string]$ev.reliability
            grounding_issues = (@($ev.grounding_issues) -join '; ')
            language_ok = [string]$ev.language_ok
            language_issue = [string]$ev.language_issue
            latency_s = [double]$record.latency_s
            error = [string]$record.error
            manual_review_required = if ($manualIds.ContainsKey([string]$record.question_id)) { 'SI' } else { 'NO' }
            manual_validation = ''
            manual_note = ''
        })
    }

    $s = $evaluation.summary
    $routing = @($evaluation.records | Group-Object tool_selection_status)
    function Count-Routing([string]$Name) {
        $item = $routing | Where-Object Name -eq $Name | Select-Object -First 1
        if ($item) { return [int]$item.Count }
        return 0
    }
    $summary.Add([pscustomobject]@{
        model = $evaluation.model
        think = 'true'
        questions = [int]$evaluation.questions_loaded
        grounded = [int]$s.reliability_breakdown.grounded
        ungrounded = [int]$s.reliability_breakdown.ungrounded
        unanswered = [int]$s.reliability_breakdown.unanswered
        unverified_no_tool = [int]$s.reliability_breakdown.unverified_no_tool
        grounded_rate = [double]$s.grounded_rate_pct / 100
        routing_optimal = Count-Routing 'optimal'
        routing_acceptable = Count-Routing 'acceptable'
        routing_suboptimal = Count-Routing 'suboptimal'
        no_tool = Count-Routing 'no_tool'
        avg_tool_calls = [double]$s.average_tool_calls_per_question
        avg_latency_s = [double]$s.average_latency_s
        language_mismatches = [string]$s.language_mismatches
        errors = [int]$s.errors
    })
}

$toolRows = @(
    @('get_scene_statistics','Sintesi quantitativa della scena e del grafo','Descrizione generale e statistiche complessive'),
    @('list_semantic_labels','Classi semantiche presenti e assenti','Domande su quali classi sono presenti o assenti'),
    @('count_objects','Conteggio totale o di una singola classe','Domande di quantita su una sola classe'),
    @('count_objects_by_class','Conteggio raggruppato per una o piu classi','Conteggi multi-classe e distribuzione'),
    @('list_objects','Elenco delle istanze per classe','Domande su quali oggetti sono presenti'),
    @('get_object_info','Geometria, posizione e dimensioni','Coordinate, centroide, bounding box e misure'),
    @('find_relationships','Relazioni di oggetti o classi specifiche','Domande focalizzate su elementi determinati'),
    @('list_relationships','Inventario complessivo delle relazioni','Riepiloghi globali del grafo'),
    @('get_object_annotation','Annotazione originale proveniente dal CSV','Descrizioni e metadati curatorali'),
    @('get_object_semantic_details','Materiale, tipologia, funzione e descrizione CSV','Domande semantiche strutturate'),
    @('find_objects_by_material','Ricerca nei soli campi materici del CSV','Domande scene-wide su uno specifico materiale')
)

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {
    $book = $excel.Workbooks.Add()
    while ($book.Worksheets.Count -lt 3) { $book.Worksheets.Add() | Out-Null }
    $wsSummary = $book.Worksheets.Item(1); $wsSummary.Name = 'Sintesi'
    $wsDetail = $book.Worksheets.Item(2); $wsDetail.Name = 'Dettaglio benchmark'
    $wsTools = $book.Worksheets.Item(3); $wsTools.Name = 'Catalogo tool'
    while ($book.Worksheets.Count -gt 3) { $book.Worksheets.Item($book.Worksheets.Count).Delete() }

    $navy = 0x5F3A16
    $teal = 0x877F1B
    $light = 0xF2EBE2
    $green = 0xD9EAD3
    $yellow = 0xD9EAFB

    $wsSummary.Cells.Item(1,1) = 'Arch-Agent - Benchmark scena4_VAL'
    $wsSummary.Cells.Item(2,1) = 'Esecuzione 17/09/2026 | 60 domande | think=true | distance threshold=2.0 m'
    $wsSummary.Range('A1:O1').Merge(); $wsSummary.Range('A2:O2').Merge()
    $wsSummary.Range('A1:O1').Interior.Color = $navy; $wsSummary.Range('A1:O1').Font.Color = 0xFFFFFF
    $wsSummary.Range('A1:O1').Font.Bold = $true; $wsSummary.Range('A1:O1').Font.Size = 18
    $wsSummary.Range('A2:O2').Interior.Color = $light; $wsSummary.Range('A2:O2').Font.Italic = $true
    $sumHeaders = @('Modello','Think','Domande','Grounded','Ungrounded','Unanswered','No tool verificabile','Grounded rate','Routing ottimale','Accettabile','Subottimale','Senza tool','Tool call medie','Latenza media (s)','Mismatch lingua')
    for($c=0;$c -lt $sumHeaders.Count;$c++){$wsSummary.Cells.Item(4,$c+1)=$sumHeaders[$c]}
    $wsSummary.Columns.Item('B').NumberFormat = '@'
    $wsSummary.Columns.Item('O').NumberFormat = '@'
    $row=5
    foreach($s in $summary){
        $vals=@($s.model,"think=$($s.think)",$s.questions,$s.grounded,$s.ungrounded,$s.unanswered,$s.unverified_no_tool,$s.grounded_rate,$s.routing_optimal,$s.routing_acceptable,$s.routing_suboptimal,$s.no_tool,$s.avg_tool_calls,$s.avg_latency_s,"$($s.language_mismatches)")
        for($c=0;$c -lt $vals.Count;$c++){$wsSummary.Cells.Item($row,$c+1)=$vals[$c]}
        $row++
    }
    $wsSummary.Range('A4:O4').Interior.Color=$teal; $wsSummary.Range('A4:O4').Font.Color=0xFFFFFF; $wsSummary.Range('A4:O4').Font.Bold=$true
    $wsSummary.Range("H5:H$($row-1)").NumberFormat='0.0%'
    $wsSummary.Range("M5:N$($row-1)").NumberFormatLocal='0,00'
    $wsSummary.Range("A4:O$($row-1)").Borders.LineStyle=1
    $wsSummary.Range("A4:O$($row-1)").AutoFilter() | Out-Null
    $wsSummary.Columns.AutoFit() | Out-Null
    $wsSummary.Columns.Item('A').ColumnWidth=18
    $wsSummary.Activate(); $excel.ActiveWindow.SplitRow=4; $excel.ActiveWindow.FreezePanes=$true

    $headers=@('Modello','Think','Question ID','Domanda','Lingua attesa','Risposta finale','Tool utilizzati','Argomenti tool','Output tool','N. tool call','Tool previsto','Tool accettabili','Routing','Problema routing','Validazione automatica','Affidabilita','Problemi grounding','Lingua OK','Problema lingua','Latenza (s)','Errore','Revisione manuale richiesta','Validazione manuale','Nota valutazione manuale')
    for($c=0;$c -lt $headers.Count;$c++){$wsDetail.Cells.Item(1,$c+1)=$headers[$c]}
    $wsDetail.Columns.Item('B').NumberFormat = '@'
    $row=2
    foreach($d in $detail){
        $vals=@($d.model,"think=$($d.think)",$d.question_id,$d.question,$d.expected_language,$d.final_answer,$d.tool_calls,$d.tool_args,$d.tool_outputs,$d.tool_call_count,$d.expected_tools,$d.acceptable_tools,$d.routing_status,$d.routing_issue,$d.automatic_validation,$d.reliability,$d.grounding_issues,$d.language_ok,$d.language_issue,$d.latency_s,$d.error,$d.manual_review_required,$d.manual_validation,$d.manual_note)
        for($c=0;$c -lt $vals.Count;$c++){$wsDetail.Cells.Item($row,$c+1)=$vals[$c]}
        $row++
    }
    $last=$row-1
    $wsDetail.Range("A1:X1").Interior.Color=$navy; $wsDetail.Range("A1:X1").Font.Color=0xFFFFFF; $wsDetail.Range("A1:X1").Font.Bold=$true
    $wsDetail.Range("A1:X$last").VerticalAlignment=-4160
    $wsDetail.Range("A1:X$last").WrapText=$true
    $wsDetail.Range("A1:X$last").Borders.Color=0xD9D9D9
    $wsDetail.Range("A1:X$last").AutoFilter() | Out-Null
    $widths=@(16,8,10,42,12,65,28,40,60,12,28,30,14,35,20,18,40,12,30,12,24,20,20,45)
    for($c=0;$c -lt $widths.Count;$c++){$wsDetail.Columns.Item($c+1).ColumnWidth=$widths[$c]}
    $wsDetail.Rows.Item(1).RowHeight=32
    $wsDetail.Range("T2:T$last").NumberFormatLocal='0,000'
    $manualRange=$wsDetail.Range("W2:W$last")
    $manualRange.Validation.Delete()
    $manualRange.Validation.Add(3,1,1,'APPROVATA,NON APPROVATA,DA REVISIONARE')
    $wsDetail.Activate(); $excel.ActiveWindow.SplitRow=1; $excel.ActiveWindow.SplitColumn=3; $excel.ActiveWindow.FreezePanes=$true

    $wsTools.Cells.Item(1,1)='Tool'; $wsTools.Cells.Item(1,2)='Descrizione'; $wsTools.Cells.Item(1,3)='Uso previsto'
    $row=2
    foreach($t in $toolRows){for($c=0;$c -lt 3;$c++){$wsTools.Cells.Item($row,$c+1)=$t[$c]};$row++}
    $wsTools.Range("A1:C1").Interior.Color=$teal; $wsTools.Range("A1:C1").Font.Color=0xFFFFFF; $wsTools.Range("A1:C1").Font.Bold=$true
    $wsTools.Range("A1:C$($row-1)").Borders.LineStyle=1; $wsTools.Range("A1:C$($row-1)").WrapText=$true
    $wsTools.Columns.Item(1).ColumnWidth=32; $wsTools.Columns.Item(2).ColumnWidth=55; $wsTools.Columns.Item(3).ColumnWidth=55
    $wsTools.Rows.AutoFit() | Out-Null

    $wsSummary.Activate()
    $book.SaveAs($OutputPath,51)
    $book.Close($true)
} finally {
    $excel.Quit()
    [Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null
}

Write-Output "Rows: $($detail.Count)"
Write-Output $OutputPath
