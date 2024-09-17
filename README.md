# Carbon Cast - A Carbon Emission Predictor 💨 

## Introduction 
By Justin:

This project uses 6 years of historical data from (Scope 1, Scope 2 and Scope 3) to predict forecast values of carbon emission to 2050. For Scope 2, the values used are based off market-based. The data is extracted from Meta, Fujitsu, Amazon, Google and Google sustainability report. It is targeted to users, company or anyone who has interest in looking at the forecasted values of carbon emission using of Scope 1, 2 and 3. It also allow users to upload their csv data file of historical data to predict (with one of my trained model) the forecasted values up to 2050.

</br>

## About this Data

This projects uses Scope 1, 2 and 3 carbon missions downloaded from the sustainability reports from the respective companies (Meta, Google, Amazon, Fujitsu, Microsoft). 

</br>

## Streamlit Demo

<h3>Demo is can be found over <a href = "https://carbon-cast.streamlit.app/">here</a></h3>
<img src="images/carboncast.jpg" alt="Image Description"></img>


## Limitations and Future Works

Currently, our model is trained on 6 years of historical data for Scope 1, 2 and 3 carbon emissions. For the model to have more accurate prediction, we would require a minimum of 10-20 years of historical data to forecast accurarely into 2050.

Users that upload their csv data are now being trained on the existing model, to forecast the output. However, if the user csv data are trained based off their own data, the forecasted values will be more accurate and precise.

I would like to incorporate carbon tax, weather & climate data, and real-time emission trends to develop a more accurate and actionable climate change mitigation strategy, enabling informed decision-making and effective policy implementation.

## Conclusion

Despite some limitation, this project serves as a proof-of-concept of its ability to forecast emission values. Addressing its limitation and incorporate future works will defintely bring the application to the next level and the world will defintely need a carbon-emission predictor like mine.

