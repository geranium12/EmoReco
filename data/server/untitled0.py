# library
import matplotlib.pyplot as plt
 
# create data
size_of_groups=[3,1]
 
# Create a pieplot
plt.pie(size_of_groups)
#plt.show()
 
# add a circle at the center
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
 
plt.show()

        '''
        m_brushTool = new BrushTool("Slice brush", this);
        m_brushTool->setBrush(m_slice->originalBrush());
        m_brush.setColor(dialog.selectedColor());
        m_slice->setBrush(m_brushTool->brush());
        
        Series->setHoleSize(0.35);
        series->append("Protein 4.2%", 4.2);
        QPieSlice *slice = series->append("Fat 15.6%", 15.6);
        slice->setExploded();
        slice->setLabelVisible();
        series->append("Other 23.8%", 23.8);
        series->append("Carbs 56.4%", 56.4);
        
        QChartView *chartView = new QChartView();
        chartView->setRenderHint(QPainter::Antialiasing);
        chartView->chart()->setTitle("Donut with a lemon glaze (100g)");
        chartView->chart()->addSeries(series);
        chartView->chart()->legend()->setAlignment(Qt::AlignBottom);
        chartView->chart()->setTheme(QChart::ChartThemeBlueCerulean);
        chartView->chart()->legend()->setFont(QFont("Arial", 7));'''
        
        
        
#QChart::ChartTheme theme = static_cast<QChart::ChartTheme>(m_themeComboBox->itemData(
 #               m_themeComboBox->currentIndex()).toInt());
   # m_chartView->chart()->setTheme(theme);
        
    #m_startAngle = new QDoubleSpinBox();
        
   # m_startAngle->setValue(m_series->pieStartAngle());
        
    #_series->setPieEndAngle(m_endAngle->value());
    
    '''m_slice->setExploded(m_sliceExploded->isChecked());
    
    CustomSlice::CustomSlice(QString label, qreal value)
    : QPieSlice(label, value)
{
    connect(this, &CustomSlice::hovered, this, &CustomSlice::showHighlight);
}

QBrush CustomSlice::originalBrush()
{
    return m_originalBrush;
}

void CustomSlice::showHighlight(bool show)
{
    if (show) {
        QBrush brush = this->brush();
        m_originalBrush = brush;
        brush.setColor(brush.color().lighter());
        setBrush(brush);
    } else {
        setBrush(m_originalBrush);
    }
}
    
    *m_series << new CustomSlice("Slice 1", 10.0);'''
    