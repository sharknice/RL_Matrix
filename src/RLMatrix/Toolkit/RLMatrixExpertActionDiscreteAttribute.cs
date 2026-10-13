using System;

namespace RLMatrix.Toolkit
{
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public class RLMatrixExpertActionDiscreteAttribute : Attribute
    {
        public int Size { get; }

        public RLMatrixExpertActionDiscreteAttribute(int size)
        {
            Size = size;
        }
    }
}
