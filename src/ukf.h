#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <utility>
#include <tuple>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::pair;
using std::tuple;

class UKF {
public:
    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    void init(MeasurementPackage meas_package);

    VectorXd getState() const;

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);

    /**
 * Computes the sigma points of the non-augmented state vector; useful for testing, not
 * actually used by the UKF.
 * @return a amtrix whose columns are the sigma points.
 */
    MatrixXd computeSigmaPoints() const;

    /**
     * Determines the augmented sigma points based on current state, covariance, noise standard
     * deviation for longitudinal acceleration and yaw acceleration.
     * @return A matrix whose columns are the augmented sigma points
     */
    MatrixXd computeAugmentedSigmaPoints() const;

    /**
     * Predicts values for given augmented sigma points after a time interval.
     * @param XsigAug the augmented sigma points, at the beginning of the time interval.
     * @param deltaT the time interval.
     * @return a matrix whose columns are the predicted sigma points.
     */
    MatrixXd predictSigmaPoints(const MatrixXd &XsigAug, double deltaT) const;

    /**
     * Determines the expected state and covariance based on given predicted sigma points.
     * @param XsigPred the predicted sigma points. The method updates the
     * object data members x and P.
     * @return a pair whose first element is the expected state vector, and second element
     * is the covariance matrix.
     */
    pair<VectorXd, MatrixXd> predictStateAndCovariance(const MatrixXd &XsigPred);

    /**
     * Determines the expected value of the next measurements, based on given predicted sigma
     * points in measurements space, and the sensor noise covariance.
     * @param Zsig a matrix whose columns are the predicted sigma points, expressed in
     * the measurements space.
     * @param R the sensor noise covariance matrix.
     * @return a pair whose first element is a vector with the measurements expected value, and
     * whose second element is its covariance matrix.
     */


    /**
 * Determines the expected value of the next measurements, based on given predicted sigma
 * points in measurements space, and the sensor noise covariance.
 * @param Zsig a matrix whose columns are the predicted sigma points, expressed in
 * the measurements space.
 * @param R the sensor noise covariance matrix.
 * @return a pair whose first element is a vector with the measurements expected value, and
 * whose second element is its covariance matrix.
 */
    pair<VectorXd, MatrixXd> predictMeasurements(const MatrixXd &Zsig, const MatrixXd &R) const;

    /**
     * Determines the expected value of the next radar measurements, based on given predicted
     * sigma points.
     * @param XsigPred a matrix whose columns are the predicted sigma points.
     * @return a triple: the first element is the vector of measurements expected values;
     * the second element is a matrix whose columns are the sigma points in measurements
     * space; the third element is the expected values covariance matrix.
     */
    tuple<VectorXd, MatrixXd, MatrixXd> predictRadarMeasurments(const MatrixXd &XsigPred) const;

    /**
     * Determines the expected value of the next lidar measurements, based on given predicted
     * sigma points.
     * @param XsigPred a matrix whose columns are the predicted sigma points.
     * @return a triple: the first element is the vector of measurements expected values;
     * the second element is a matrix whose columns are the sigma points in measurements
     * space; the third element is the expected values covariance matrix.
     */
    tuple<VectorXd, MatrixXd, MatrixXd> predictLidarMeasurments(const MatrixXd &XsigPred) const;

    /**
     * Updates the currently predicted state x, and its covariance P, based on the latest sensor measurments.
     * Also computes the NIS. The method updates the object data members x, P and nis.
     * @param meas_package the latest sensor measurements, to be used for update.
     * @param zPred the measurements expected value (predicted value), as returned by predictLidarMeasurments() or
     * predictRadarMeasurments().
     * @param Zsig the sigma points in measurements space, as returned by predictLidarMeasurments() or
     * predictRadarMeasurments().
     * @param S the covariance for the measurements expected value (predicted value), as returned by
     * predictLidarMeasurments() or predictRadarMeasurments().
     * @param XsigPred the predicted sigma points, as returned by predictSigmaPoints().
     * @return a triple: the first element is the updated state vector; the second element is the updated state
     * covariance; the third is the computed NIS.
     */
    tuple<VectorXd, MatrixXd, double>
    updateStateWithMeasurements(const MeasurementPackage &meas_package, const VectorXd &zPred, const MatrixXd &Zsig,
                                const MatrixXd &S, const MatrixXd &XsigPred);


    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

    // state covariance matrix
    Eigen::MatrixXd P_;

    // predicted sigma points matrix
    Eigen::MatrixXd Xsig_pred_;

    // time when the state is true, in us
    long long time_us_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    // Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    // Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    // Radar measurement noise standard deviation radius in m
    double std_radr_;

    // Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    // Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    // Weights of sigma points
    // Eigen::VectorXd weights_;

    // State dimension
    int n_x_;

    // Augmented state dimension
    int n_aug_;

    // Number of dimension in augmented state vector
    int xAug_n;

    // Sigma point spreading parameter
    double lambda_;

    MatrixXd XsigPred;


    /* Weights entering the prediction of the state and its covariance, the prediction of measurements
    and their covariance, and calculation of the cross-correlation matrix. */
    VectorXd weights_;

    // The last computed NIS (defaults to 0 if not yet computed)
    double nis;

};

#endif  // UKF_H