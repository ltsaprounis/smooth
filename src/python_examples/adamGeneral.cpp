#include <iostream>
#include <cmath>

#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <adamGeneral.h>

namespace py = pybind11;

// # Fitter for univariate models
// Convert from Rcpp::List to pybind11::dict

py::dict adamFitter(arma::mat &matrixVt, arma::mat const &matrixWt, arma::mat &matrixF, arma::vec const &vectorG,
                    arma::uvec &lags, arma::umat const &indexLookupTable, arma::mat profilesRecent,
                    char const &E, char const &T, char const &S,
                    unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                    unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                    arma::vec const &vectorYt, arma::vec const &vectorOt, bool const &backcast)
{
    /* # matrixVt should have a length of obs + lagsModelMax.
     * # matrixWt is a matrix with nrows = obs
     * # vecG should be a vector
     * # lags is a vector of lags
     */

    int obs = vectorYt.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = matrixVt.n_rows;
    int lagsModelMax = max(lags);

    // Fitted values and the residuals
    arma::vec vecYfit(obs, arma::fill::zeros);
    arma::vec vecErrors(obs, arma::fill::zeros);

    // Loop for the backcasting
    unsigned int nIterations = 1;
    if (backcast)
    {
        nIterations = 2;
    }

    // Loop for the backcast
    for (unsigned int j = 1; j <= nIterations; j = j + 1)
    {

        // Refine the head (in order for it to make sense)
        // This is only needed for ETS(*,Z,*) models, with trend.
        // if(!backcast){
        for (int i = 0; i < lagsModelMax; i = i + 1)
        {
            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                 matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
        }
        // }
        ////// Run forward
        // Loop for the model construction
        for (int i = lagsModelMax; i < obs + lagsModelMax; i = i + 1)
        {

            /* # Measurement equation and the error term */
            vecYfit(i - lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                                                   matrixWt.row(i - lagsModelMax), E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            // If this is zero (intermittent), then set error to zero
            if (vectorOt(i - lagsModelMax) == 0)
            {
                vecErrors(i - lagsModelMax) = 0;
            }
            else
            {
                vecErrors(i - lagsModelMax) = errorf(vectorYt(i - lagsModelMax), vecYfit(i - lagsModelMax), E);
            }

            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                 matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                      adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(i - lagsModelMax), E, T, S,
                                                                 nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant, vectorG, vecErrors(i - lagsModelMax));

            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
        }

        ////// Backwards run
        if (backcast && j < (nIterations))
        {
            // Change the specific element in the state vector to negative
            if (T == 'A')
            {
                profilesRecent(1) = -profilesRecent(1);
            }
            else if (T == 'M')
            {
                profilesRecent(1) = 1 / profilesRecent(1);
            }

            for (int i = obs + lagsModelMax - 1; i >= lagsModelMax; i = i - 1)
            {
                /* # Measurement equation and the error term */
                vecYfit(i - lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                                                       matrixWt.row(i - lagsModelMax), E, T, S,
                                                       nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // If this is zero (intermittent), then set error to zero
                if (vectorOt(i - lagsModelMax) == 0)
                {
                    vecErrors(i - lagsModelMax) = 0;
                }
                else
                {
                    vecErrors(i - lagsModelMax) = errorf(vectorYt(i - lagsModelMax), vecYfit(i - lagsModelMax), E);
                }

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                          adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF,
                                                                     matrixWt.row(i - lagsModelMax), E, T, S,
                                                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                                                     vectorG, vecErrors(i - lagsModelMax));
            }

            // Fill in the head of the series
            for (int i = lagsModelMax - 1; i >= 0; i = i - 1)
            {
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);

                matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
            }

            // Change back the specific element in the state vector
            if (T == 'A')
            {
                profilesRecent(1) = -profilesRecent(1);
            }
            else if (T == 'M')
            {
                profilesRecent(1) = 1 / profilesRecent(1);
            }
        }
    }

    // return List::create(Named("matVt") = matrixVt, Named("yFitted") = vecYfit,
    //                     Named("errors") = vecErrors, Named("profile") = profilesRecent);

    // Create a Python dictionary to return results
    py::dict result;
    result["matVt"] = matrixVt;
    result["yFitted"] = vecYfit;
    result["errors"] = vecErrors;
    result["profile"] = profilesRecent;

    return result;
}

/* # Function produces the point forecasts for the specified model */
arma::vec adamForecaster(arma::mat const &matrixWt, arma::mat const &matrixF,
                         arma::uvec lags, arma::umat const &indexLookupTable, arma::mat profilesRecent,
                         char const &E, char const &T, char const &S,
                         unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                         unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                         unsigned int const &horizon)
{
    // unsigned int lagslength = lags.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    unsigned int nComponents = indexLookupTable.n_rows;

    arma::vec vecYfor(horizon, arma::fill::zeros);

    /* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i = 0; i < horizon; i = i + 1)
    {
        vecYfor.row(i) = adamWvalue(profilesRecent(indexLookupTable.col(i)), matrixWt.row(i), E, T, S,
                                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

        profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                             matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
    }

    // return List::create(Named("matVt") = matrixVtnew, Named("yForecast") = vecYfor);
    return vecYfor;
}

py::dict adamPolynomialiser(arma::vec const &B,
                                   arma::uvec const &arOrders, arma::uvec const &iOrders, arma::uvec const &maOrders,
                                   bool const &arEstimate, bool const &maEstimate,
                                   arma::vec armaParametersValue, arma::uvec const &lags){

    // Sometimes armaParameters is NULL. Treat this correctly
    // arma::vec armaParametersValue;
    // if(!Rf_isNull(armaParameters)){
    //     armaParametersValue = as<arma::vec>(armaParameters);
    // }

    // TODO:
    // Removing SEXP armaParameters and replacing it with arma::vec armaParametersValue
    // is a quick hack to avoid the Rcpp type SEXP that was used here previously
    // The value armaParameters can be either arma::vec or Null and we currently don't 
    // support Null.

// Form matrices with parameters, that are then used for polynomial multiplication
    arma::mat arParameters(max(arOrders % lags)+1, arOrders.n_elem, arma::fill::zeros);
    arma::mat iParameters(max(iOrders % lags)+1, iOrders.n_elem, arma::fill::zeros);
    arma::mat maParameters(max(maOrders % lags)+1, maOrders.n_elem, arma::fill::zeros);

    arParameters.row(0).fill(1);
    iParameters.row(0).fill(1);
    maParameters.row(0).fill(1);

    int lagsModelMax = max(lags);

    int nParam = 0;
    int armanParam = 0;
    for(unsigned int i=0; i<lags.n_rows; ++i){
        if(arOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<arOrders(i); ++j){
                if(arEstimate){
                    arParameters((j+1)*lags(i),i) = -B(nParam);
                    nParam += 1;
                }
                else{
                    arParameters((j+1)*lags(i),i) = -armaParametersValue(armanParam);
                    armanParam += 1;
                }
            }
        }

        if(iOrders(i) * lags(i) != 0){
            iParameters(lags(i),i) = -1;
        }

        if(maOrders(i) * lags(i) != 0){
            for(unsigned int j=0; j<maOrders(i); ++j){
                if(maEstimate){
                    maParameters((j+1)*lags(i),i) = B(nParam);
                    nParam += 1;
                }
                else{
                    maParameters((j+1)*lags(i),i) = armaParametersValue(armanParam);
                    armanParam += 1;
                }
            }
        }
    }

// Prepare vectors with coefficients for polynomials
    arma::vec arPolynomial(sum(arOrders % lags)+1, arma::fill::zeros);
    arma::vec iPolynomial(sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec maPolynomial(sum(maOrders % lags)+1, arma::fill::zeros);
    arma::vec ariPolynomial(sum(arOrders % lags)+sum(iOrders % lags)+1, arma::fill::zeros);
    arma::vec bufferPolynomial;

    arPolynomial.rows(0,arOrders(0)*lags(0)) = arParameters.submat(0,0,arOrders(0)*lags(0),0);
    iPolynomial.rows(0,iOrders(0)*lags(0)) = iParameters.submat(0,0,iOrders(0)*lags(0),0);
    maPolynomial.rows(0,maOrders(0)*lags(0)) = maParameters.submat(0,0,maOrders(0)*lags(0),0);

    for(unsigned int i=0; i<lags.n_rows; ++i){
// Form polynomials
        if(i!=0){
            bufferPolynomial = polyMult(arPolynomial, arParameters.col(i));
            arPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

            bufferPolynomial = polyMult(maPolynomial, maParameters.col(i));
            maPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

            bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
            iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
        }
        if(iOrders(i)>1){
            for(unsigned int j=1; j<iOrders(i); ++j){
                bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
                iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
            }
        }

    }
    // ariPolynomial contains 1 in the first place
    ariPolynomial = polyMult(arPolynomial, iPolynomial);

    // Check if the length of polynomials is correct. Fix if needed
    // This might happen if one of parameters became equal to zero
    if(maPolynomial.n_rows!=sum(maOrders % lags)+1){
        maPolynomial.resize(sum(maOrders % lags)+1);
    }
    if(ariPolynomial.n_rows!=sum(arOrders % lags)+sum(iOrders % lags)+1){
        ariPolynomial.resize(sum(arOrders % lags)+sum(iOrders % lags)+1);
    }
    if(arPolynomial.n_rows!=sum(arOrders % lags)+1){
        arPolynomial.resize(sum(arOrders % lags)+1);
    }

    // return wrap(List::create(Named("arPolynomial") = arPolynomial, Named("iPolynomial") = iPolynomial,
    //                          Named("ariPolynomial") = ariPolynomial, Named("maPolynomial") = maPolynomial));
    // Create a Python dictionary to return results
    py::dict result;
    result["arPolynomial"] = arPolynomial;
    result["iPolynomial"] = iPolynomial;
    result["ariPolynomial"] = ariPolynomial;
    result["maPolynomial"] = maPolynomial;

    return result;
}

PYBIND11_MODULE(_adam_general, m)
{
    m.doc() = "Adam code"; // module docstring
    m.def(
        "adam_fitter",
        &adamFitter,
        "fits the adam model",
        py::arg("matrixVt"),
        py::arg("matrixWt"),
        py::arg("matrixF"),
        py::arg("vectorG"),
        py::arg("lags"),
        py::arg("indexLookupTable"),
        py::arg("profilesRecent"),
        py::arg("E"),
        py::arg("T"),
        py::arg("S"),
        py::arg("nNonSeasonal"),
        py::arg("nSeasonal"),
        py::arg("nArima"),
        py::arg("nXreg"),
        py::arg("constant"),
        py::arg("vectorYt"),
        py::arg("vectorOt"),
        py::arg("backcast"));
    m.def(
        "adam_forecaster",
        &adamForecaster,
        "forecasts the adam model",
        py::arg("matrixWt"),
        py::arg("matrixF"),
        py::arg("lags"),
        py::arg("indexLookupTable"),
        py::arg("profilesRecent"),
        py::arg("E"),
        py::arg("T"),
        py::arg("S"),
        py::arg("nNonSeasonal"),
        py::arg("nSeasonal"),
        py::arg("nArima"),
        py::arg("nXreg"),
        py::arg("constant"),
        py::arg("horizon"));
    m.def(
        "adam_polynomializer",
        &adamPolynomialiser,
        "Adam polynomials for different orders",
        py::arg("B"),
        py::arg("arOrders"),
        py::arg("iOrders"),
        py::arg("maOrders"),
        py::arg("arEstimate"),
        py::arg("maEstimate"),
        py::arg("armaParametersValue"),
        py::arg("lags"));
}

