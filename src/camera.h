#ifndef CAMERA_H_
#define CAMERA_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <GL/gl.h>
#include <GL/glu.h>

#define PI 3.1415926535897932384626433832795
#define PIdiv180 (PI/180.0)

/**
 * @class Camera
 * @brief Simple camera class similar to the ones used in first-person shooter video games.
 *        Based on the OpenGL coordinate system and gluLookAt().
 * @note There is a problem when upVector is colinear to viewVector.
 *       To prevent the so called <em> gimbal lock </em>, use a camera class based on Quaternions instead.
 */
class Camera {

public:

	/**
	 * Delegating constructor
	 * @param x,y,z Coordinates of the initial position in 3D space.
	 */
	Camera(double x, double y, double z);

	/**
	 * Delegating constructor
	 * @param pos Coordinate vector of the initial position in 3D space.
	 */
	Camera(Eigen::Vector3d pos);

	/**
	 * Delegating constructor
	 * @param pos Coordinate vector of the initial position in 3D space.
	 * @param view Initial view vector.
	 */
	Camera(Eigen::Vector3d pos, Eigen::Vector3d view);

	virtual ~Camera();

	/**
	 * (Re)sets the current position vector.
	 * @param pos Coordinate vector of the position in 3D space.
	 * @return void
	 */
	void setPosition(Eigen::Vector3d pos);

	/**
	 * (Re)sets the current view vector.
	 * @param view View vector.
	 * @return void
	 */
	void setView(Eigen::Vector3d view);

	/**
	 * Returns the current position vector.
	 * @return The position vector.
	 */
	Eigen::Vector3d getPosition() { return positionVector; };

	/**
	 * Returns the current view vector.
	 * @return The view vector.
	 */
	Eigen::Vector3d getView() { return viewVector; };

	/**
	 * Computes the new pitch, i.e. the rotation around the rightVector (look up/down).
	 * @note: We won't compute the new up vector since this may lead to a gimbal lock.
	 * @param angle The rotation angle.
	 * @return void
	 */
	void rotateX(double angle);

	/**
	 * Computes the new Yaw, i.e. the rotation around the upVector (look left/right).
	 * @param angle The rotation angle.
	 * @return void
	 */
	void rotateY(double angle);

	/**
	 * Moves sideways and computes the new position vector.
	 * @param distance Distance in x-direction (OpenGL coordinate system)
	 * @return void
	 */
	void moveX(double distance);

	/**
	 * Moves forward/backward and computes the new position vector.
	 * @param distance Distance in y-direction (OpenGL coordinate system)
	 * @return void
	 */
	void moveY(double distance);

	/**
	 * Moves upward/downward and computes the new position vector.
	 * @param distance Distance in z-direction (OpenGL coordinate system)
	 * @return void
	 */
	void moveZ(double distance);

	/**
	 * Computes the new position vector based on the given direction vector.
	 * @param direction Distance in xyz-direction (OpenGL coordinate system)
	 * @return void
	 */
	void move(Eigen::Vector3d& direction);

	/**
	 * Final step: executes gluLookAt().
	 * @return void
	 */
	void setup();

private:

	/**
	 * The delegating constructors point to this initialization function.
	 */
	void init(Eigen::Vector3d pos, Eigen::Vector3d view);

	float speedFactor;

	Eigen::Vector3d positionVector;
	Eigen::Vector3d viewVector;
	Eigen::Vector3d rightVector;
	Eigen::Vector3d upVector;
	Eigen::Vector3d targetPosition;
};

#endif /* CAMERA_H_ */
