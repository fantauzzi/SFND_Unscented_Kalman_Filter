#include "tools.h"
#include "ukf.h"
#include "Eigen/Dense"
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::tuple;
using std::make_tuple;
using std::make_pair;
using std::get;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() : is_initialized_{false},
             use_laser_{true},
             use_radar_{true},
             x_{VectorXd(5)},
             P_{MatrixXd(5, 5)},
             std_a_{5},
             std_yawdd_{.85},
             n_x_{static_cast<int>(x_.size())},
             xAug_n{n_x_ + 2} {

    // Initialise weights_, done here once and for all.
    auto lambda = 3. - xAug_n;
    weights_ = VectorXd(2 * xAug_n + 1);
    weights_.setZero();
    weights_(0) = lambda / (lambda + xAug_n);  // Set the first component
    weights_.bottomRows(2 * xAug_n).setConstant(1 / (2 * (lambda + xAug_n))); // Set the remaining 2*n_aug components

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */

}

void UKF::init(MeasurementPackage meas_package) {
    assert(!is_initialized_); // Member function should be called only once
    is_initialized_ = true;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        auto px = meas_package.raw_measurements_(0);
        auto py = meas_package.raw_measurements_(1);
        auto v = 0.;
        auto psi = 0.;
        auto psiDot = 0.;
        x_ << px, py, v, psi, psiDot;
    } else {
        assert(meas_package.sensor_type_ == MeasurementPackage::RADAR);
        auto rho = meas_package.raw_measurements_(0);
        auto theta = meas_package.raw_measurements_(1);
        auto px = rho * cos(theta);
        auto py = rho * sin(theta);
        auto v = 0.;
        auto psi = 0.;
        auto psiDot = 0.;
        x_ << px, py, v, psi, psiDot;
    }

    // Initialise current state covariance
    P_.setZero(); // It works better than identity matrix
}


MatrixXd UKF::computeAugmentedSigmaPoints() const {
    // The augmented state
    VectorXd xAug = VectorXd(xAug_n);
    xAug.setZero();
    xAug.head(n_x_) = x_;

    // Covariance matrix for acceleration and yaw acceleration errors
    auto Q = MatrixXd(2, 2);
    Q << pow(std_a_, 2), 0, 0, pow(std_yawdd_, 2);

    // The augmented state covariance
    MatrixXd P_Aug = MatrixXd(xAug_n, xAug_n);
    P_Aug.setZero();
    P_Aug.block(0, 0, n_x_, n_x_) = P_;
    P_Aug.block(n_x_, n_x_, 2, 2) = Q;

    // Get the square root of the augmented state covariance
    MatrixXd A_Aug = P_Aug.llt().matrixL();

    //Calculate augmented sigma points
    MatrixXd XsigAug = MatrixXd(xAug_n, 2 * xAug_n + 1);
    auto lambda = 3 - xAug_n;
    auto spreadAug = sqrt(lambda + xAug_n);
    XsigAug.col(0) = xAug;
    for (auto iCol = 0; iCol < xAug_n; ++iCol) {
        XsigAug.col(iCol + 1) = xAug + spreadAug * A_Aug.col(iCol);
        XsigAug.col(iCol + 1 + xAug_n) = xAug - spreadAug * A_Aug.col(iCol);
    }

    return XsigAug;
}

MatrixXd UKF::predictSigmaPoints(const MatrixXd &XsigAug, double deltaT) const {
    double epsilon = 0.000001;  // Threshold under which absolute value of psiDot is considered 0

    //Make matrix with predicted sigma points as columns
    auto XsigPred = MatrixXd(n_x_, 2 * xAug_n + 1);
    for (auto i = 0; i < 2 * xAug_n + 1; ++i) {
        auto v = XsigAug(2, i);
        auto psi = XsigAug(3, i);
        auto psiDot = XsigAug(4, i);
        auto nu_a = XsigAug(5, i);
        auto nu_psiDotDot = XsigAug(6, i);
        VectorXd b{VectorXd(5)};
        b << .5 * pow(deltaT, 2) * cos(psi) * nu_a, .5 * pow(deltaT, 2)
                                                    * sin(psi) * nu_a, deltaT * nu_a, .5 * pow(deltaT, 2)
                                                                                      * nu_psiDotDot, deltaT *
                                                                                                      nu_psiDotDot;
        VectorXd a{VectorXd(5)};
        if (abs(psiDot) < epsilon)
            a << v * cos(psi) * deltaT, v * sin(psi) * deltaT, 0, psiDot
                                                                  * deltaT, 0;
        else
            a << v / psiDot * (sin(psi + psiDot * deltaT) - sin(psi)), v
                                                                       / psiDot *
                                                                       (-cos(psi + psiDot * deltaT) + cos(psi)), 0,
                    psiDot
                    * deltaT, 0;
        XsigPred.col(i) = XsigAug.block(0, i, 5, 1) + a + b;
    }

    return XsigPred;
}

pair<VectorXd, MatrixXd> UKF::predictStateAndCovariance(
        const MatrixXd &XsigPred) {
    // Determine the predicted state x based on predicted sigma-points and pre-computed weights_
    x_.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i)
        x_ += weights_(i) * XsigPred.col(i);
    x_(3) = normaliseAngle(x_(3));

    // Determine the covariance P of the predicted state
    P_.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; i++) {  //iterate over sigma points
        // State difference
        VectorXd x_diff = XsigPred.col(i) - x_;
        // Angle normalisation
        x_diff(3) = normaliseAngle(x_diff(3));
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }

    // Return predicted state x and its covariance P
    auto ret = make_pair(x_, P_);
    return ret;
}


void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {

    if (!is_initialized_) {
        init(meas_package);
        time_us_ = meas_package.timestamp_;
        return;
    }

    /** If the measurement is from an instrument to be ignored, do nothing and just return.
    * Note: no update to time_us_ at this time.
    */

    if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
        return;

    if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
        return;

    // Calculate the time elapsed between the previous measurement (weather lidar or radar) and the current one,
    // in seconds.
    double deltaT = static_cast<double>(meas_package.timestamp_ - time_us_) / 1000000.0; // seconds

    /*
     * Prediction step
     */

    auto XsigPred = Prediction(deltaT);

    /*
     * Update step
     */

    auto predictedMeasurements =
            (meas_package.sensor_type_ == MeasurementPackage::RADAR) ?
            predictRadarMeasurments(XsigPred) :
            predictLidarMeasurments(XsigPred);

    // Fetch the necessary to update the current state based on the latest measurement, and compute the NIS
    auto zPred{get<0>(predictedMeasurements)};
    auto Zsig{get<1>(predictedMeasurements)};
    auto S{get<2>(predictedMeasurements)};
    auto res = updateStateWithMeasurements(meas_package, zPred, Zsig, S,
                                           XsigPred);

    // Update the UKF instance NIS
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        nis_radar = get<2>(res);
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        nis_lidar = get<2>(res);

    // Now it's the time to update previousTimeStamp, to be used in the next iteration
    time_us_ = meas_package.timestamp_;
}

tuple<VectorXd, MatrixXd, MatrixXd> UKF::predictRadarMeasurments(
        const MatrixXd &XsigPred_radar) const {
    const int z_n = 3; // Number of components in radar measurements

    // Will store sigma points in measurement space
    MatrixXd Zsig = MatrixXd(z_n, 2 * xAug_n + 1);

    // Transform sigma points into measurement space, implementing the radar measurement model
    Zsig.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i) {
        auto px = XsigPred_radar(0, i);
        auto py = XsigPred_radar(1, i);
        auto v = XsigPred_radar(2, i);
        auto psi = XsigPred_radar(3, i);
        auto rho = sqrt(px * px + py * py);
        auto phi = atan2(py, px);
        auto rhoDot = (px * cos(psi) * v + py * sin(psi) * v) / rho;
        Zsig.col(i) << rho, phi, rhoDot;
    }

    // Radar noise covariance
    auto R = MatrixXd(3, 3);
    R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_ * std_radrd_;

    auto predicted = predictMeasurements(Zsig, R);
    auto zPred = predicted.first;  // The predicted measurement points
    auto S = predicted.second;  // Their covariance

    return make_tuple(zPred, Zsig, S);
}

tuple<VectorXd, MatrixXd, MatrixXd> UKF::predictLidarMeasurments(
        const MatrixXd &XsigPred_lidar) const {
    const int z_n = 2; // Number of components in lidar measurments

    // Will store sigma points in measurement space
    MatrixXd Zsig = MatrixXd(z_n, 2 * xAug_n + 1);

    // Transform sigma points into measurement space, implementing the lidar measurement model
    Zsig.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i) {
        auto px = XsigPred_lidar(0, i);
        auto py = XsigPred_lidar(1, i);
        Zsig.col(i) << px, py;
    }

    // Lidar noise covariance TODO could this be moved to ctor or init()?
    auto R = MatrixXd(2, 2);
    R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

    auto predicted = predictMeasurements(Zsig, R);
    auto zPred = predicted.first;  // The predicted measurement points
    auto S = predicted.second;  // Their covariance

    return make_tuple(zPred, Zsig, S);
}

tuple<VectorXd, MatrixXd, double> UKF::updateStateWithMeasurements(
        const MeasurementPackage &meas_package, const VectorXd &zPred,
        const MatrixXd &Zsig, const MatrixXd &S, const MatrixXd &XsigPred) {
    // Make a matrix to store cross correlation, Tc
    VectorXd z{meas_package.raw_measurements_};
    const auto z_n = z.size();
    MatrixXd Tc = MatrixXd(n_x_, z_n);

    // Calculate cross correlation matrix, Tc
    Tc.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i) {
        VectorXd x_diff = XsigPred.col(i) - x_;
        x_diff(3) = normaliseAngle(x_diff(3));
        VectorXd z_diff = Zsig.col(i) - zPred;
        if (z_diff.size() ==
            3) // If z_dif has 3 dimensions then measurement comes from radar, and angle theta needs to be normalised
            z_diff(1) = normaliseAngle(z_diff(1));
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    // Calculate Kalman gain K;
    MatrixXd K{Tc * S.inverse()};

    // Update state mean x and its covariance P
    VectorXd z_diff = z - zPred;
    if (z_diff.size() ==
        3) // If z_dif has 3 dimensions then measurement comes from radar, and angle theta needs to be normalised
        z_diff(1) = normaliseAngle(z_diff(1));
    x_ += K * z_diff;
    x_(3) = normaliseAngle(x_(3));
    P_ -= K * S * K.transpose();

    // Finally compute the NIS
    double nis = z_diff.transpose() * S.inverse() * z_diff;

    return make_tuple(x_, P_, nis);
}


pair<VectorXd, MatrixXd> UKF::predictMeasurements(const MatrixXd &Zsig,
                                                  const MatrixXd &R) const {

    // Determine the measurement expected value (predicted measurement) and store it in zPred
    const auto z_n = Zsig.rows();
    VectorXd zPred = VectorXd(z_n);
    zPred.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i)
        zPred += weights_(i) * Zsig.col(i);

    // Determine the predicted measurement covariance, and store it in S
    auto S = MatrixXd(z_n, z_n);
    S.setZero();
    for (auto i = 0; i < 2 * xAug_n + 1; ++i) {
        VectorXd zDiff = Zsig.col(i) - zPred;
        zDiff(1) = normaliseAngle(zDiff(1));
        S += weights_(i) * (zDiff * zDiff.transpose());
    }
    S += R;

    // Return predicted measurement and its covariance
    return make_pair(zPred, S);
}


MatrixXd UKF::Prediction(double delta_t) {
    assert(is_initialized_);

    // Determine the sigma points of the previous belief
    auto XsigAug = computeAugmentedSigmaPoints();
    // Propagate the sigma points through the noise-free state prediction
    auto XsigPred = predictSigmaPoints(XsigAug, delta_t);
    // Predict mean and covariance based on the predicted sigma points
    predictStateAndCovariance(XsigPred);

    return XsigPred;
}
