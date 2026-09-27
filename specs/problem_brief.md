# Problem Brief -- Smelter Super Kings

KPI:                      reduce surface-related rejection rate
Business problem:         overall surface rejection increased from 2.3% to 3.6%, mainly driven by stain defects increasing from 0.4% to 1.4%
ML problem:               using the captured inspection image and product type available at first inspection, classify each aluminium product as acceptable, scratch, dent, stain, or other
Goal:                     help the quality team identify stain reliably and consistently, separate it from other surface defects, and trigger faster investigation
ML type:                  multiclass image classification
Target classes:           acceptable, scratch, dent, stain, other
Inputs known at prediction time: inspection image, product type
Business success metric:  faster and more consistent stain identification, enabling quicker investigation of upstream causes
Model success metric:     macro F1 overall, with stain recall and stain F1 treated as priority metrics
Track:                    Image (Surface Defect)
Data used in class:       SYNTHETIC
