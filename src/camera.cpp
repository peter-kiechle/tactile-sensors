#include <cstdio>
#include <iostream>

#include "camera.h"

Camera::Camera(double x, double y, double z) {
	init(Eigen::Vector3d(x, y, z), Eigen::Vector3d(0.0, 0.0, -1.0) );
}

Camera::Camera(Eigen::Vector3d pos) {
	init(pos, Eigen::Vector3d(0.0, 0.0, -1.0) );
}

Camera::Camera(Eigen::Vector3d pos, Eigen::Vector3d view) {
	init(pos, view);
}

void Camera::init(Eigen::Vector3d pos, Eigen::Vector3d view) {
	// Camera speed
	speedFactor = 1.5f;

	// Coordinate system with standard OpenGL values:
	positionVector = pos;
	viewVector = view;
	rightVector = Eigen::Vector3d(1.0, 0.0, 0.0);
	upVector = Eigen::Vector3d(0.0, 1.0, 0.0);
}

Camera::~Camera() {}

void Camera::setPosition(Eigen::Vector3d pos) {
	positionVector = pos;
}

void Camera::setView(Eigen::Vector3d view) {
	viewVector = view;
}

void Camera::rotateX (double angle) {
	// Compute new view-vector
	viewVector = viewVector * cos(angle*PIdiv180) + upVector * sin(angle*PIdiv180);
	viewVector.normalize();
}

void Camera::rotateY(double angle) {
	// Compute new view-vector
	viewVector = viewVector*cos(angle*PIdiv180) - rightVector*sin(angle*PIdiv180);
	viewVector.normalize();

	// Compute the new right-vector by cross product
	rightVector = viewVector.cross(upVector);
	rightVector.normalize();
}

void Camera::moveX(double distance) {
	positionVector = positionVector + (rightVector *(distance * speedFactor));
}

void Camera::moveY(double distance) {
	positionVector = positionVector + (upVector * (distance * speedFactor));
}

void Camera::moveZ(double distance) {
	positionVector = positionVector + (viewVector * (distance * speedFactor));
}

void Camera::move(Eigen::Vector3d& direction) {
	positionVector = positionVector + direction;
}

void Camera::setup() {
	// The positionVector the camera aims at
	targetPosition = positionVector + viewVector;

	//printf("\nPosition: % 2.3f, % 2.3f, % 2.3f\n", positionVector[0], positionVector[1], positionVector[2]);
	//printf("View:     % 2.3f, % 2.3f, % 2.3f\n", viewVector[0], viewVector[1], viewVector[2]);

	gluLookAt(positionVector.x(), positionVector.y(), positionVector.z(),
			  targetPosition.x(), targetPosition.y(), targetPosition.z(),
			  upVector.x(), upVector.y(), upVector.z()); // Workaround: hard-coded to (0,1,0)
}
