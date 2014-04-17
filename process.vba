Function IsInArray(ByVal stringToBeFound As String, arr As Variant) As Boolean
  IsInArray = (UBound(Filter(arr, stringToBeFound)) > -1)
End Function


Sub GraphSheets(worksheetGroup() As Worksheet, ByVal repeats As Integer, ByVal title As String)
'
' Plot a log graph of a de run.
' worksheetGroup may contain empty elements!
'
    For Each ws In worksheetGroup
        If Not ws Is Nothing Then
            ' Ensure we're on a valid active worksheet
            ws.Activate
        End If
    Next ws
    ActiveSheet.Shapes.AddChart.Select
    ' Define the data
    ActiveChart.ChartType = xlXYScatterLinesNoMarkers
    ActiveChart.Location Where:=xlLocationAsNewSheet
    ActiveChart.Name = title
    ActiveChart.Tab.Color = 192
    For Each s In ActiveChart.SeriesCollection
        s.Delete
    Next
    For Each ws In worksheetGroup
        If Not ws Is Nothing Then
            ' Ensure we're on a valid active worksheet
            Set cell1 = ws.Cells(1, (repeats * 4) + 1)
            Set xr = ws.Range(cell1, ws.Cells(cell1.End(xlDown), (repeats * 4) + 1))
            Set yr = ws.Range(ws.Cells(1, (repeats * 4) + 2), ws.Cells(cell1.End(xlDown), (repeats * 4) + 2))
            With ActiveChart.SeriesCollection.NewSeries
                .Name = ws.Name
                .XValues = xr
                .Values = yr
            End With
        End If
    Next ws
    ' Title
    ActiveChart.HasTitle = False
    ' ActiveChart.ChartTitle.Characters.Text = title
    ' X axis
    ActiveChart.Axes(xlCategory, xlPrimary).HasTitle = True
    ActiveChart.Axes(xlCategory, xlPrimary).AxisTitle.Characters.Text = "Function Evaluations"
    ActiveChart.Axes(xlCategory).TickLabels.NumberFormat = "0.0E+00"
    ' Y axis
    ActiveChart.Axes(xlValue, xlPrimary).HasTitle = True
    ActiveChart.Axes(xlValue, xlPrimary).AxisTitle.Characters.Text = "Function Value"
    ActiveChart.Axes(xlValue).MajorGridlines.Delete
    ActiveChart.Axes(xlValue).ScaleType = xlLogarithmic
    ActiveChart.Axes(xlValue).MinimumScale = 0.000001
    ActiveChart.Axes(xlValue).CrossesAt = 0.000001
    ActiveChart.Axes(xlValue).MajorUnit = 100
    ActiveChart.Axes(xlValue).TickLabels.NumberFormat = "0.E+00"
    ActiveChart.Move After:=Sheets(Sheets.Count)
    ' Assume four plots
    ActiveChart.SeriesCollection(2).Select
    With Selection.Format.Line
        .Visible = msoTrue
        .DashStyle = msoLineSysDot
    End With
    ActiveChart.SeriesCollection(3).Select
    With Selection.Format.Line
        .Visible = msoTrue
        .DashStyle = msoLineDash
    End With
    ActiveChart.SeriesCollection(4).Select
    With Selection.Format.Line
        .Visible = msoTrue
        .DashStyle = msoLineDashDot
    End With
    
End Sub



Sub CalculateMedians(ByVal repeats As Integer)
    Dim vals() As Double
    Dim r As Range
    Dim ws As Worksheet

    For Each ws In ActiveWorkbook.Worksheets
            
        ws.UsedRange.NumberFormat = "0.00E+00"
        
        For j = 1 To ws.UsedRange.Rows.Count
        
            For k = 1 To 4
                
                ReDim vals(repeats - 1)
                For l = 0 To (repeats - 1)
                    
                    Set workingCell = ws.Cells(j, k + (4 * l))
                    If IsEmpty(workingCell) Then
                        vals(l) = workingCell.End(xlUp).Value
                    Else
                        vals(l) = workingCell.Value
                    End If
                Next l

                ws.Cells(j, k + (repeats * 4)).Value = WorksheetFunction.Median(vals)
                ws.Cells(j, k + (repeats * 4)).Font.Bold = True
                
                If k = 2 Then
                    ' Compute Mean, standard dev and success %
                    ws.Cells(j, 5 + (repeats * 4)).Value = WorksheetFunction.Average(vals)
                    ws.Cells(j, 5 + (repeats * 4)).Font.Bold = True
                    ws.Cells(j, 6 + (repeats * 4)).Value = WorksheetFunction.StDev(vals)
                    ws.Cells(j, 6 + (repeats * 4)).Font.Bold = True
                    c = 0
                    For Each v In vals
                        If v <= 0.000001 Then c = c + 1
                    Next v
                    ws.Cells(j, 7 + (repeats * 4)).Value = (c * 100) / repeats
                    ws.Cells(j, 7 + (repeats * 4)).Font.Bold = True
                End If
                    
            Next k

        Next j
        
    Next ws
    
End Sub


Sub GroupSheets(ByVal repeats As Integer)

    ' First, group sheets by category
    Dim wsarray() As String
    ReDim wsarray(1 To Worksheets.Count)
    Dim titlearray() As String
    j = 1
    For i = 1 To Worksheets.Count
        titlearray = Split(Worksheets(i).Name, "_")
        If UBound(titlearray) > 0 Then
            x = titlearray(0)
            Debug.Print x
            If IsInArray(x, wsarray) = False Then
                wsarray(j) = x
                j = j + 1
            Else
                wsarray(j) = ""
            End If
        End If
    Next i
    
    ' Second, loop over the categories to assemble groups.
    Dim worksheetGroup() As Worksheet
    j = 1
    For Each shtName In wsarray
        ' Doubles as an erase function.
        ReDim worksheetGroup(1 To Worksheets.Count)
        If shtName <> "" Then
            For Each ws In ActiveWorkbook.Worksheets
                If InStr(1, ws.Name, shtName) > 0 Then
                    Set worksheetGroup(j) = ws
                    j = j + 1
                End If
            Next ws
            ' Call the GraphSheets function after each group has been assembled
            Call GraphSheets(worksheetGroup, repeats, shtName)
        End If
    Next shtName
    
End Sub

Sub MakeTables()
'
' Create mean, standard dev and success tables
'
Dim ws As Worksheet

Worksheets.Add(After:=Worksheets(Worksheets.Count)).Name = "Tables"
ActiveSheet.Tab.Color = 10
Set ts = Worksheets("Tables")
ts.Columns("B").ColumnWidth = ts.Columns("B").ColumnWidth * 2
i = 1
For Each ws In ActiveWorkbook.Worksheets
    titlearray = Split(ws.Name, "_")
    If UBound(titlearray) > 0 Then
        ts.Cells(i, 1).Value = titlearray(0)
        ts.Cells(i, 2).Value = titlearray(1)
        cc = ws.UsedRange.Columns.Count
        rc = ws.UsedRange.Rows.Count
        ts.Cells(i, 3).Value = ws.Cells(rc, cc - 2).Value
        ts.Cells(i, 4).Value = ws.Cells(rc, cc - 1).Value
        ts.Cells(i, 5).Value = ws.Cells(rc, cc).Value
        i = i + 1
    End If
Next

End Sub

Sub Macro4()
'
' Macro4 Macro
'

'
For Each cht In ActiveWorkbook.Charts
    cht.Axes(xlValue).AxisTitle.Font.Size = 12
    cht.Axes(xlCategory).AxisTitle.Font.Size = 12
    cht.Axes(xlValue).TickLabels.Font.Size = 12
    cht.Axes(xlCategory).TickLabels.Font.Size = 12
    cht.Legend.Font.Size = 12
    cht.Legend.IncludeInLayout = False
Next cht

End Sub



Sub main()
'
' Process the spreadsheet in full.
'

Application.ScreenUpdating = False
'Check if any sheets are charts.
cCount = 0
For Each oCs In ActiveWorkbook.Charts
    cCount = cCount + 1
Next
used = ActiveWorkbook.ActiveSheet.UsedRange.Columns.Count
If cCount > 0 Then used = used - 7
repeats = used / 4
Call CalculateMedians(repeats)
If cCount = 0 Then Call GroupSheets(repeats)
Call MakeTables
Application.ScreenUpdating = True

End Sub


