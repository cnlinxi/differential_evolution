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
    ActiveChart.HasTitle = True
    ActiveChart.ChartTitle.Characters.Text = title
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
    
End Sub



Sub CalculateMedians(ByVal repeats As Integer)

    Dim vals() As Double
    Dim r As Range
    Dim ws As Worksheet

    ' Compute Median values
    For Each ws In ActiveWorkbook.Worksheets
            
        ws.UsedRange.NumberFormat = "0.00E+00"
        
        For j = 1 To ws.UsedRange.Rows.Count
            
            For k = 1 To 4
            
                ReDim vals(repeats)
            
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
            x = titlearray(0) & "_" & titlearray(1)
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


Sub main()
'
' Process the spreadsheet in full.
'
repeats = ActiveWorkbook.ActiveSheet.UsedRange.Columns.Count / 4
Call CalculateMedians(repeats)
Call GroupSheets(repeats)

End Sub
