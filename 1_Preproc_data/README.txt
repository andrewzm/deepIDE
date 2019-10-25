How to obtain the SST data in the ./data folder
###############################################

1. Visit:
http://marine.copernicus.eu/services-portfolio/technical-faq/ and look for the "GLOBAL-ANALYSIS-FORECAST-PHY-001-024" under the FAQ question that contains the word "Archive". Alternatively, visit 
http://nrtcmems.mercator-ocean.fr/motu-web/Motu?action=productdownloadhome&service=GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS&product=global-analysis-forecast-phy-001-024
Note that you will need Copernicus login details to access the site.

2. Set the spatial window to
-70 to -20 lon
33 to 62 lat

and the temporal window to 2006-12-27 -- 2019-03-12

3. Change select depth to 0.49 - 0.50 m

4. Choose only SST to download (theta0)

5. Click download


Running the Python program
##########################

Run visdom server in a separate terminal
python -m visdom.server
