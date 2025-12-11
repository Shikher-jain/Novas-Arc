from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

model = SentenceTransformer('all-MiniLM-L6-v2')

model_text= """
An Amazon Virtual Private Cloud (VPC) is a virtual network dedicated to your AWS account. It is logically isolated from other virtual networks in the AWS cloud. You can launch your AWS resources, such as Amazon EC2 instances, into your VPC. You can specify an IP address range for the VPC, add subnets, associate security groups, and configure route tables. A subnet is a range of IP addresses in your VPC. You can launch AWS resources into a specified subnet. Use a public subnet for resources that must be connected to the Internet, and use a private subnet for resources that won't be connected to the Internet. A VPC spans all of the Availability Zones in the region. After creating a VPC, you can add one or more subnets in each Availability Zone. When you create a subnet, you specify the CIDR block for the subnet, which is a subset of the VPC CIDR block. The VPC CIDR block plus the CIDR blocks of all the subnets cannot overlap with each other or with an on-premises network. You can also optionally specify an IPv6 CIDR block for the VPC.
"""
site_text = """
Amazon VPC lets you provision a logically isolated section of the Amazon Web Services (AWS) cloud where you can launch AWS resources in a virtual network that you define. You have complete control over your virtual networking environment, including selection of your own IP address ranges, creation of subnets, and configuration of route tables and network gateways. You can also create a hardware Virtual Private Network (VPN) connection between your corporate datacenter and your VPC and leverage the AWS cloud as an extension of your corporate datacenter.
You can easily customize the network configuration for your Amazon VPC. For example, you can create a public-facing subnet for your web servers that have access to the Internet, and place your backend systems such as databases or application servers in a private-facing subnet with no Internet access. You can leverage multiple layers of security, including security groups and network access control lists, to help control access to Amazon EC2 instances in each subnet.
"""

embeddings = model.encode([model_text, site_text])
# print(embeddings)
similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

print(f"Semantic Similarity: {similarity:.2f}")

if similarity > 0.80:
    print("Same meaning.")
elif similarity > 0.6:
    print("Related.")
else:
    print("Different meaning.")

