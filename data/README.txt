**************Information on how to use our Multi-layout Invoice Document Dataset (MIDD) : A Dataset for Named Entity Recognition************

Folder name: MIDD IOB 
Inside this folder there are Four Layout Invoices IOB from four different suppliers.
IOB files are transformed into .csv extension so that you can use it readily for model training.

Layout 1:    196  .csv files
Layout 2:      29  .csv files
Layout 3:     14  .csv files
Layout 4:    391  .csv files

Total           630 .csv files

The labels used for annotations are:
***********************************************Key Fields Annotation names*******************************
*****Supp_N for Supplier name************  Key Field
*****Supp_G for Supplier GST number****** Key Field
*****BUY_N for Buyer Name************** Key Field
*****BUY_G for Buyer GST number******** Key Field
*****INV_NO for Invoice Number********** Key Field
*****INV_DT for Invoice Date************* Key Field
*****GT_AMT for Grand Total Amount******Key Field.
***********************************************************Label******************************************
These labels are used to denote a string used to show the key fields.
For example (Invoice No: 2778888),so Invoice No is a label with INV_L and its actual value is INV_NO.
Similar for other fields.

INV_L for Invoice Number Label, 
INV_DL for Invoice Date Label,
GT_AMTL for Grand Total Amount Label, 
GSTL for GST Label. 

In each IOB ,there are two columns:
First Column: Every word or token from invoice.
Second Column: Token's or word's I or O or B tag.




