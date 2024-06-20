
# Python Virtual Environment Setup Guide for macOS

## Step 1: Install Python
Ensure Python is installed on your machine. You can download it from the [official Python website](https://www.python.org/downloads/). During installation, make sure to check the option to add Python to your PATH.

## Step 2: Install `virtualenv`
`virtualenv` is a tool to create isolated Python environments. It’s a best practice to use it for project-specific dependencies.

```sh
pip install virtualenv
```

## Step 3: Create a Virtual Environment
Navigate to your project directory and create a virtual environment. Replace `myenv` with your preferred environment name.

```sh
cd /path/to/your/project
virtualenv myenv
```

## Step 4: Activate the Virtual Environment
Activate the virtual environment using the following command:

```sh
source myenv/bin/activate
```

After activation, you should see the virtual environment name prefixed in your terminal prompt, indicating that the virtual environment is active.

## Step 5: Install Dependencies from `requirements.txt`
Ensure you have a `requirements.txt` file in your project directory. This file should list all the required libraries for your project.

```sh
pip install -r requirements.txt
```

## Step 6: Verify the Installation
You can check if the libraries were installed correctly by listing the installed packages.

```sh
pip list
```

Following these steps will help you manage project-specific dependencies effectively, avoiding conflicts with global Python packages.
