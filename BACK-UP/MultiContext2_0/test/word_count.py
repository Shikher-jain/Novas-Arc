txt ='''
### 🧭 **Domain:** AWS Account Setup
### 🗣️ **Persona:** Fresher
### 🎯 **Intent:** Setup AWS Account
### 📚 **Tone:** Detailed
---

To set up an AWS account, you need to perform the following steps:

1. **Sign Up for AWS**
    - Open the Amazon Web Services home page.
    - Choose **Create an AWS Account**.
    - Enter your account information, and then choose **Continue**.
    - Choose **Personal** or **Professional**.
    - Enter your contact information, and then choose **Continue**.
    - Enter your payment information.
    - Read and accept the AWS Customer Agreement.
    - Choose **Create Account and Continue**.

2. **Create an IAM User**
    - Open the Amazon Web Services home page.
    - Choose **Sign in to the Console**.
    - Choose **I am a returning user and my password is**.
    - Enter your email address and password.
    - Choose **Sign in using our secure server**.
    - In the navigation pane, choose **Users**.
    - Choose **Add user**.
    - Enter user information.
    - Choose **Next: Permissions**.
    - Choose **Attach existing policies directly**.
    - In the search box, type **AdministratorAccess**.
    - Select the check box next to **AdministratorAccess**.
    - Choose **Next: Tags**.
    - Choose **Next: Review**.
    - Choose **Create user**.

3. **Sign In to the Console**
    - Open the Amazon Web Services home page.
    - Choose **Sign in to the Console**.
    - Enter your email address and password.
    - Choose **Sign in using our secure server**.

4. **Secure Your Account**
    - In the navigation pane, choose **IAM**.
    - Choose **Activate MFA on your root account**.
    - Follow the instructions to enable MFA on your root account.
    - In the navigation pane, choose **Dashboard**.
    - Choose **Customize**.
    - Choose **Enable Billing Alerts**.
    - Choose **Save preferences**.

5. **Set Up Billing Alerts**
    - In the navigation pane, choose **Billing & Cost Management**.
    - Choose **Billing Preferences**.
    - Choose **Receive Billing Alerts**.
    - Choose **Save preferences**.

6. **Create an S3 Bucket**
    - In the navigation pane, choose **Services**.
    - Choose **S3**.
    - Choose **Create bucket**.
    - Enter a unique bucket name.
    - Choose the Region where you want the bucket to reside.
    - Choose **Create bucket**.

7. **Launch an EC2 Instance**
    - In the navigation pane, choose **Services**.
    - Choose **EC2**.
    - Choose **Launch instance**.
    - Choose an Amazon Machine Image (AMI).
    - Choose an Instance Type.
    - Choose **Review and Launch**.
    - Choose **Launch**.

8. **Terminate Your Resources**
    - In the navigation pane, choose **Services**.
    - Choose **EC2**.
    - Select the instance, choose **Actions**, choose **Instance State**, and then choose **Terminate**.
    - Choose **Terminate**.

9. **Close Your AWS Account**
    - Open the Amazon Web Services home page.
    - Choose **Sign in to the Console**.
    - Enter your email address and password.
    - Choose **Sign in using our secure server**.
    - In the navigation pane, choose **Support**.
    - Choose **Support Center**.
    - Choose **Create case**.
    - Choose **Account and Billing Support**.
    - Choose **Close my account**.
    - Choose **Email**.

10. **Get Help**
    - In the navigation pane, choose **Support**.
    - Choose **Support Center**.
    - Choose **Create case**.
    - Choose **Service limit increase**.
    - Choose **EC2 Instances**.
    - Choose **Request limit increase**.
'''
print(len(txt.split(" ")))